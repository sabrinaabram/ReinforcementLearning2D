% Created 16 January 2018
% The purpose of the first version of this code is to combine Max's
% 2D reinforcement learning algorithm with recent advances in Jess' 
% 1D reinforcement learning algorithm

clear
close all

%% Max's 2D RL
% Here we will use Mark Snaterse's speed/frequency treadmill experiment to
% start thinking about energy optimization as reinforcement learning in 2
% dimensions. The end goal is to extend this thinking to width/frequency.

% Marks experiment used a control function to control treadmill speed based
% on people's step frequency. He finds that people
% rapidly converge to the stable intersection of their normally preferred
% relationship between speed and step frequency and the relationship between
% step frequency and speed enforced by the treadmill control function. He also
% finds that over the course of his experimental protocol, people stay at the
% intersection rather than find the new energetic minimum that does not lie at
% the intersection. (Snaterse et al, 2011).

% This is a simple reinforcement learning model that first rapidly
% converges to a stable fixed point, and then slowly converges to the new
% global energy optimum.

%% Load Cost Contour Data from Max
% This data is generated in a separate m-file called CostContours.m. It uses
% data that we collected in our own lab (Jess' JAP data) and combines it with
% Umberger's step frequency data to approximate gross COT as a function of both
% speed and step frequency. I then fit a model to that simulated data, and then
% use the model to predict cost at gridded combinations of frequency and speed.
load('CostContourData');

% S: Speed grid from 0.75 - 1.75 m/s in 0.01 m/s increments (dif freqs are rows and dif speeds are columns) 
% F: Step Freq grid from 66-147 spm in 1 spm increments (dif freqs are rows and dif speeds are columns) 
% E: gross COT normalized for min value which occurs at 1.32 m/s and 107 spm in this smoothed data.
% sp: a vector of all the speeds
% fp: a vector of the preferred freq at each of the speeds. 
% Note that speeds are rounded to the closest 0.01 and frequencies to the nearest 1 spm
S = round(100*S)/100; % this is necessary because some values are not exactly at certain speeds

% This is to test the dependence of convergence rate of steepness of cost curve.
% SABRINA: I do not understand this.
if 0
    addedcost = [fliplr(linspace(0,1,floor(size(S,2)/2))) linspace(0,1,ceil(size(S,2)/2))];
    E = E+repmat(addedcost,size(S,1),1);
end

% Define natural cost landscape (function of both freq and speed).
% This is variable E from loaded 'CostContourData'
Qa_nat = E;

% The range of preferred frequencies for certain speeds are stored in fp (frequency preferred)
% We want to find the energetic costs for these frequencies. To
% do this, we need to first find where this frequency is in F, and then
% find the energy of that point in Qa_nat, or E.
for i=1:length(sp) % sp is a vector of all the speeds, 0.75 - 1.75 m/s
    F_fpref = find(F(:,i)==fp(i));
    E_fpref(i) = Qa_nat(F_fpref,i); % energy of preferred frequency at each speed
end

% plot the speed/frequency/energy 2D relationship, as well as the preferred
% relationship between speed and frequency.
figure(1); clf; 
subplot(2,1,1); hold on
h = contour(S,F,Qa_nat,[1.01 1.05:0.05:1.5],'ShowText','On');
plot(sp,fp,'k')
title('Controller Off Gross COT (as fraction of minimum)')
ylabel('Step Frequency (spm)')
xlabel('Speed (m/s)')

% find the energy minimum of this landscape and plot.
[~,ind] = min(Qa_nat(:));
[freq_natEmin,spd_natEmin]=ind2sub(size(Qa_nat),ind);
plot(S(freq_natEmin,spd_natEmin),F(freq_natEmin,spd_natEmin),'r.','MarkerSize',30)

%% Control Function
% Max:
% The way I did this was to look at the cost contour plot from the above code
% and just choose a location on the preferred s-f curve that occurs at a high
% speed and cost. I then eyeballed a lower speed and step frequency pair that
% gives the control function a slope that is greater than the pref relationship
% (and thus stable) while also yielding a minimum cost along the control
% function that occurs at a different speed and step frequency than what is
% normally minimal.

% SABRINA: need to think more about why a greater slope is stable

x1 = 1.3; y1 = 100; % first pt on ctrl function
x2 = 1.7; y2 = 121; % second pt on ctrl function

% solve a system of two equations for slope and intercept parameters
x = [x1;x2];
y = [y1;y2];
X = [x1 1;x2 1];
ctrlfunc = inv(X)*y; % ctrlfunc(1) is slope, ctrlfunc(2) is intercept

% visualize control function.
% x is speed, y is frequency
x_spd = sp(1):0.01:sp(end);
y_freq = ctrlfunc(1)*x_spd + ctrlfunc(2);
plot(x_spd,y_freq,'b')

% Similar to above, find the energetic costs of frequencies in control
% function.
for i=1:length(sp)
    F_freqcf = find(F(:,i)==round(y_freq(i)));
    % Create Qa_shift (actual cost with controller on)
    E_freqcf(i) = Qa_nat(F_freqcf,i);
end

% Determine where the cost min lies along control function
for i=1:length(x_spd)
    xi(i) = find(S(1,:) == round(100*x_spd(i))/100);
    yi(i) = find(F(:,1) == round(y_freq(i)));    
    cfE(i) = Qa_nat(yi(i),xi(i));
end
[~, cfEmin] = min(cfE);

%% Max's take on 2D learning:
% The way I have come to think about this is as a grid of states. The columns of
% this grid are the speed states and the rows are the step frequency states. 
%
% At each speed state, the only actions that the person can take is to change
% step frequency as treadmill speed changes are out of their control. One can
% think about this as actions that take them up and down the grid boxes to dif
% freqs, but left and right actions to different treadmill speeds are not
% allowed. They are allowed to take any sized actions up and down the grid
% spaces - they can go from any step frequency to any other step frequency.
%
% Unlike many of the Sutton and Barto examples, this grid has dynamics. So when
% a person makes an up or down movement along the grid, the treadmill can pull
% them left or right to a different state then perhaps they were targetting
% (although they don't actually target states in this RI framework).
%
% Initially, subjects find themselves at a certain speed. They then choose their
% action by searching for the minimum expected cost at that speed. Their
% action-value function maps all step frequencies to expected values at that
% speed. 
% 
% Once they choose a frequency/action, the treadmill/environment shoots them to
% a new speed and the reward they get is the cost at the step frequency they
% choose and the speed that the control function chooses for them.
%
% Their initial action-value function is the cost contour where they assume that 
% the speed at which they choose their freq is also the speed at which they will
% get their reward. The action-value function that they learn is the cost that
% the will experience at the speed where the treadmill will take them based on
% their chosen action/frequency. 

%% Define characteristics of learning agent
% Below is Jess' simple RL algorithm with reference cost implemented in a
% 2D learning space

% execution noise (step frequency, or action, variabilty)
exec_noise=1.5; % spm

% measurement noise (cost variabilty)
meas_noise=0.02; 

% weighting of new measurements (learning rate)
alpha=0.5; 

% criteria for initiation of optimization (step when begin optimizing)
% set spont = 1 for spontaneous initiation of optimization
% set spont = 0 for non-spontaneous initiation of optimization
spont = 0;

if spont == 1
    % for spontaneous initaitors
    % int_opt=361; % this is simply set to change at when controller turns on
    int_opt=1; % this is simply set always be optimizing
else
    % for non-spontaneous initiators
    % before the hold low, ony update reference cost
    % after hold low, update all costs
    int_opt=1440; % this is simply set to change at beginning of first hold low
end

%% Define characteritics of protocol
experience = 1;

if experience == 1
    % for learning in this 2D landscape, we will first try giving them equal
    % experience in both dimensions. This experience will be through
    % perturbations to both higher and lower costs in both dimensions. The
    % perturbation will consist of a 3-minute hold and a 3-minute release. The
    % learner will be able to self-select actions during each release.

    twindow=3;
    % twindow minutes for each of following:

    % in natural cost landscape
    % 1. controller off, no visual feedback

    % in new 2D cost landscape
    % 2. controller on, no visual feedback
    % 3. controller on, visual feedback hold high cost SF
    % 4. controller on, no visual feedback
    % 5. controller on, visual feedback hold low cost SF
    % 6. controller on, no visual feedback
    % 7. controller on, visual feedback hold high cost SF
    % 8. controller on, no visual feedback
    % 9. controller on, visual feedback hold low cost SF
    % 10. controller on, no visual feedback
    % 11. controller on, visual feedback hold high cost SF
    % 12. controller on, no visual feedback
    % 13. controller on, visual feedback hold low cost SF
    % 14. controller on, no visual feedback

    % typical sf in Hz;
    hz=2; % just used to define number of steps in protocol
    steps=twindow*24*60*hz; % total number of steps

    % steps when controller turns on and off
    oncon=twindow*60*hz;
    
    con=zeros(steps, 1);
    con(oncon:end)=1;

    % steps when visual feedback turns on 
    hold_steps=twindow.*[2:2:24].*60*hz; % steps when hold begins
    % length that metrnome stays on 
    hold_len=twindow*60*hz; % length of hold in steps
    % assigned action when metrnome turns on 
    [~,holdhigh]=min(abs(((F(:,1)-90))));[~,holdlow]=min(abs(((F(:,1)-106))));
    speedhigh = round(100*((F(holdhigh,1) - ctrlfunc(2))/ctrlfunc(1)))/100;
    speedlow = round(100*((F(holdlow,1) - ctrlfunc(2))/ctrlfunc(1)))/100;
    
    hold_actions=repmat([holdhigh holdlow],1,24/2); 
    met=nan(steps, 1);
    met_sf=nan(steps, 1);
    for h = 1:length(hold_steps)
        met(hold_steps(h):hold_steps(h)+hold_len)=hold_actions(h);
        met_sf(hold_steps(h):hold_steps(h)+hold_len)=F(hold_actions(h),1);
    end

    % plot protocol
    % figure(2)
    % plot(con); hold on
    % xlabel('steps')
    % ylabel('0=off, 1=on')
    % ylim([-0.5 1.5])
    % xlim([0 steps])
    % plot(met_sf, 'r','LineWidth', 2)
    % xlabel('steps')
    % ylabel('metronome value')
    % ylim([-15 15])
    % xlim([0 steps])
    % xticks([0:hold_len:steps])
else
    steps = 10000; % number of steps to take in a given experiment.
    met=nan(steps, 1); % visual feedback off.
    met_sf=nan(steps, 1); % self-select steps.
    con=ones(steps, 1); % steps when controller turns on and off
end

%% Define characteristics of analysis
% number of repeats (experiments or subjects)
repeats=2;  % use for quick partial simulation
% repeats=1000;  % use for full simulation

%% Loop through subjects/experiments
for r=1:repeats
    disp(r)

    % the following is re-set for new subject
    % Qa_nat: freqs are rows and speeds are columns
    Qa_est = Qa_nat;  % inital estimate of cost in natural (will be updated)
    spdold = S(1,spd_natEmin); % old speed is the speed at which the person selects their freq/action but not the speed at which cost is evaluated when the controller is on

    % set up empty vectors
    sf_all=NaN(steps,1); % store step frequency for each step
    spd_all = NaN(steps,1); % store speed of treadmill for each step
    action_all=NaN(steps,1); % store action for each step
    reward_all=NaN(steps,1); % store reward for each step
    Qa_est_all=NaN(size(F,1),size(S,2),steps); % store current Q est for each step

    % evaluate for each step
    for s=1:steps

        % Choose Action.
        if isnan(met(s))==0 % Define actions when visual feedback on
            action = met(s);
        else % obey the policy and choose min cost when visual feedback off
            oldspdind = find(S(1,:)==spdold); % speed column
            [~, action] = min(Qa_est(:,oldspdind)); % choose freq index that has min cost at the old speed 
        end

        % add execution noise to step frequency
        % note that this is how Jess adds in the execution noise, Max does
        % this differently in MaxMarkRlSim.m
        newsf = F(action,oldspdind) + exec_noise*randn;
        
        % solve for the actual executed action
        [~,action]= min(abs(F(:,oldspdind)-newsf));

        % solve for the reward for the action
        if con(s)==0 % controller off
            newspd = S(1,oldspdind); % this is because controller off does not shoot agent to new speed
            reward = Qa_nat(action, oldspdind) + meas_noise.*randn;
        else % controller on
            % Environment shoots agent to new speed
            newspd = round(100*((newsf - ctrlfunc(2))/ctrlfunc(1)))/100; % tm changes speed based on chosen freq and control function
            if newspd>1.75, newspd = 1.75; elseif newspd <0.75; newspd =0.75; end % handle grid edges
            newspdind = find(S(1,:)==newspd); % speed index
            reward = Qa_nat(action, newspdind) + meas_noise.*randn;
        end

        % Update estimate of cost landscape based on action
        if s < int_opt % only update reference cost
            qNew = Qa_est(action,oldspdind);
            Qa_est=Qa_est + alpha*(reward-Qa_est(action,oldspdind));
            % note: max does this diferrently 
            % in his reward-Qa_est(action) is reward-Qa_est(31) # optima
        else % update entire landscape
            qOld = Qa_est(action,oldspdind); % the old expected reward based on old speed and freq
            qNew = qOld + alpha*(reward-qOld); % the new expected reward for this action is a weighted combination of old expected reward and new reward 

            % Update action-value function
            Qa_est(action,oldspdind) = qNew;            
        end

        % log data for each step  
        sf_all(s) = newsf;
        spd_all(s) = newspd;
        action_all(s) = action;
        reward_all(s) = reward;
        Qa_est_all(:,:,s) = Qa_est;
        Qai_est_all(s) = qNew;
        
        % The new treadmill speed becomes the old/current treadmill speed for
        % the subject to select their new step frequency
        spdold = newspd;
    end
    
    % calculate final preferred
    t_fin = length(sf_all);
    t_init = t_fin - 180*hz;
    fprefSf_i(r) = mean(sf_all(t_init:t_fin));
    fprefSpd_i(r) = mean(spd_all(t_init:t_fin));

    % log data across repeats
    sf_all_all(r,:)=sf_all;
    spd_all_all(r,:)=spd_all;
    action_all_all(r,:)=action_all';
    reward_all_all(r,:)=reward_all';
    Qa_est_all_all(r,:,:,:)=Qa_est_all;
    Qai_est_all_all(r,:)=Qai_est_all;
end

% average final prefs across subjects
fprefSf_avg = mean(fprefSf_i);
fprefSpd_avg = mean(fprefSpd_i);

%% Plot the outcome, averaged across subjects
figure(3)
subplot(3,1,1)
hold on
plot(mean(sf_all_all));
plot(met_sf,'r','LineWidth', 2)
xlim([0 steps])
xl = get(gca,'xlim');
yl = get(gca,'ylim');
initMin = plot(xl, F(freq_natEmin,spd_natEmin)*[1 1],'r'); % initial cost min
intPt = plot(xl, y2*[1 1],'k'); % intersection pt
newMin = plot(xl, y_freq(cfEmin)*[1 1],'g'); % final cost min along control function
if experience == 1
    plot([oncon, oncon],[yl(1) yl(2)],'--r', 'LineWidth',1)
    xticks([0:hold_len:steps])
end
xlabel('Steps')
ylabel('Step Frequency (spm)')
legend([initMin,newMin,intPt],{'initial cost min','final cost min','intersection point'})

subplot(3,1,2)
hold on
plot(mean(spd_all_all));
xlim([0 steps])
xl = get(gca,'xlim');
yl = get(gca,'ylim');
plot(xl, S(freq_natEmin,spd_natEmin)*[1 1],'r') % initial cost min
plot(xl, x2*[1 1],'k') % intersection pt
plot(xl, x_spd(cfEmin)*[1 1],'g') % final cost min along control function
if experience == 1
    plot([oncon, oncon],[yl(1) yl(2)],'--r', 'LineWidth',1)
    xticks([0:hold_len:steps])
end
xlabel('Steps')
ylabel('Speed (m/s)')

subplot(3,1,3)
hold on
Qai_mean = mean(Qai_est_all_all);
plot(Qai_mean);
% determine the cost during holds
met_E=nan(steps, 1);
for h = 1:length(hold_steps)-1
    met_Ehold = mean(Qai_mean(hold_steps(h):hold_steps(h)+hold_len));
    met_E(hold_steps(h):hold_steps(h)+hold_len)=met_Ehold;
end
plot(met_E,'r','LineWidth', 2)
ylim([0.95 1.25])
xlim([0 steps])
xl = get(gca,'xlim');
yl = get(gca,'ylim');
if experience == 1
    plot([oncon, oncon],[yl(1) yl(2)],'--r', 'LineWidth',1)
    xticks([0:hold_len:steps])
end
xlabel('Steps')
ylabel('Estimated Cost (?)')

if spont == 1 % spont 
    savefig('Fig1Ba.fig')
    save('spont_data','sf_all_all','met_sf','steps')
else % non spont
    save('nonspont_data','sf_all_all','met_sf','steps')
    savefig('Fig1Bb.fig')
end

%% Plot the estimate of the cost landscape at each time point
Qa_est_all_all_avg=squeeze(mean(Qa_est_all_all,1));
% if experience == 1
%     plt_steps=0:hold_len:steps; plt_steps(1)=1;plt_steps(2:end)=plt_steps(2:end)-1;
%     plt_steps(11)=[];
% else
%     plt_steps = round(linspace(1,10000-1,4*3));
% end
% 
% figure(4)
% for p=1:length(plt_steps)
%     subplot(4,3,p)
%     hold on
%     contour(S,F,Qa_est_all_all_avg(:,:,1),[1.01 1.05:0.05:1.5],'LineColor','r','ShowText','On');
%     contour(S,F,Qa_est_all_all_avg(:,:,plt_steps(p)),[1.01 1.05:0.05:1.5],'LineColor','b','ShowText','On');
%     xlabel('Speed (m/s)')
%     ylabel('Step Frequency (spm)')
% %     ylim([ 0 2])
% end

%% Plot learned estimate of cost landscape
figure(1)
subplot(2,1,2); hold on
contour(S,F,Qa_est_all_all_avg(:,:,end),[1.01 1.05:0.05:1.5],'ShowText','On');
origmin = plot(S(freq_natEmin,spd_natEmin),F(freq_natEmin,spd_natEmin),'r.','MarkerSize',30);
newmin = plot(x_spd(cfEmin),y_freq(cfEmin),'g.','MarkerSize',30);
finalpref = plot(fprefSpd_avg,fprefSf_avg,'b.','MarkerSize',30);
holds = plot(speedhigh,F(holdhigh,1),'k.','MarkerSize',30);
plot(speedlow,F(holdlow,1),'k.','MarkerSize',30)

legend([origmin,newmin,finalpref,holds],{'original cost min','new cost min','final preferred','holds'})

title('Controller On Gross COT (as fraction of minimum)')
ylabel('Step Frequency (spm)')
xlabel('Speed (m/s)')

%% Plot a slice of the cost landscape along the control function
figure(5)
subplot(1,2,1)
hold on
plot(x_spd,cfE)
title('slice of control function along speed dimension')
newCost = plot(x_spd(cfEmin),cfE(cfEmin),'g.','MarkerSize',30);
origpref = plot(S(freq_natEmin,spd_natEmin),E(freq_natEmin,spd_natEmin),'r');
% newpref = plot(fprefSpd_avg);
xl = get(gca,'xlim');
yl = get(gca,'ylim');
holds = plot([speedhigh speedhigh],[yl(1) yl(2)],'k--','MarkerSize',30);
holds = plot([speedlow speedlow],[yl(1) yl(2)],'k--','MarkerSize',30);
ylabel('Estimated Cost (?)')
xlabel('Speed (m/s)')

subplot(1,2,2)
hold on
plot(y_freq,cfE)
plot(y_freq(cfEmin),cfE(cfEmin),'g.','MarkerSize',30)
% origpref = plot(S(freq_natEmin,spd_natEmin),E(freq_natEmin,spd_natEmin),'r');
xl = get(gca,'xlim');
yl = get(gca,'ylim');
holds = plot([F(holdhigh,1) F(holdhigh,1)],[yl(1) yl(2)],'k--','MarkerSize',30);
holds = plot([F(holdlow,1) F(holdlow,1)],[yl(1) yl(2)],'k--','MarkerSize',30);
title('slice of control function along frequency dimension')
xlabel('Frequency (spm)')

%% Plot adaptation after holds

% if experience == 1
%     figure(100)
%     subplot(2,2,1)
%     plot(mean(sf_all_all(:,360-2*60:360+2*60))); hold on
%     plot([120,120],[-20,15],'--k'); hold on
%     plot([0,60*2*2],[0,0],'--k'); hold on
%     plot([0,60*2*2],[sf(cfEmin),sf(cfEmin)],'--k'); hold on
%     ylim([ -20, 15]); xlim([ 0, 60*2*2])
% 
%     subplot(2,2,3)
%     plot(mean(sf_all_all(:,3600-2*60:3600+2*60))); hold on
%     %plot(mean(sf_all_all(:,3120:3360))); hold on
%     plot([0,240],[0,0],'--k'); hold on
%     plot([0,240],[sf(cfEmin),sf(cfEmin)],'--k'); hold on
%     plot([120,120],[-20,15],'--k'); hold on
%     ylim([ -20, 15]); xlim([ 0, 240])
% 
%     subplot(2,2,2)
%     plot(mean(sf_all_all(:,1080-2*60:1080+2*60))); hold on
%     plot(mean(sf_all_all(:,1680:1920))); hold on
%     plot([0,240],[0,0],'--k'); hold on
%     plot([0,240],[sf(cfEmin),sf(cfEmin)],'--k'); hold on
%     ylim([ -20, 15]); xlim([ 0, 240])
%     plot([120,120],[-20,15],'--k'); hold on
%     legend('high1', 'low1')
% 
%     subplot(2,2,4)
%     plot(mean(sf_all_all(:,2520-2*60:2520+2*60))); hold on
%     plot(mean(sf_all_all(:,3120:3360))); hold on
%     legend('high2', 'low2')
%     plot([0,240],[0,0],'--k'); hold on
%     plot([0,240],[sf(cfEmin),sf(cfEmin)],'--k'); hold on
%     plot([120,120],[-20,15],'--k'); hold on
%     ylim([ -20, 15]); xlim([ 0, 240])
% 
%     % subplot(3,1,3)
%     % plot(mean(sf_all_all(:,1680:1920))); hold on
%     % plot(mean(sf_all_all(:,3120:3360))); hold on
%     % legend('low1', 'low2')
%     % plot([0,240],[0,0],'--k'); hold on
%     % plot([0,240],[-7,-7],'--k'); hold on
%     % ylim([ -20, 15]); xlim([ 0, 240])
% end