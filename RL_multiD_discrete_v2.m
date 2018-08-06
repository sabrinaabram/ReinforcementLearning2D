% Created 22 March 2018
% The purpose of this code is to extend Jess' 1D RL to multiple dimensions,
% and then test how the adaptation time constant depends on the number of
% dimensions

clear
close all
set(0,'defaultAxesFontSize',14)

%% Cost landscapes
% dimensions (can generalize to any number)
n = 2;

% speed frequency 2D relationship
% Load Cost Contour Data
% This data is generated in a separate m-file called CostContours.m. It uses
% data that we collected in our own lab (Jess' JAP data) and combines it with
% Umberger's step frequency data to approximate gross COT as a function of both
% speed and step frequency. I then fit a model to that simulated data, and then
% use the model to predict cost at gridded combinations of frequency and speed.
load('../CostContourData');
% S: Speed grid from 0.75 - 1.75 m/s in 0.01 m/s increments (dif freqs are rows and dif speeds are columns) 
% F: Step Freq grid from 66-147 spm in 1 spm increments (dif freqs are rows and dif speeds are columns) 
% E: gross COT normalized for min value which occurs at 1.32 m/s and 107 spm in this smoothed data.
% sp: a vector of all the speeds
% fp: a vector of the preferred freq at each of the speeds. 
% Note that speeds are rounded to the closest 0.01 and frequencies to the nearest 1 spm

Qa_nat = E(:);

% find the energy minimum
% frequency is rows
% speed is columns
[~,indNat] = min(Qa_nat(:));
subsNat = cell([1 n]);  % dynamically sized variable
[subsNati, subsNatj]=ind2sub(size(E),indNat);
subsNat = cell2mat(subsNat);

Freqmin = F(subsNati,1);
Speedmin = S(1,subsNatj);

Speed = ((S - Speedmin)./Speedmin).*100;
Frequency = ((F - Freqmin)./Freqmin).*100;

% load pref speed when freq constrained overground
load('../v-f_data')
% ff2 corresponds to the prescribed step frequencies during the overground constrained 
% experiments and ss corresponds to the corresponding steady-state speeds.

% choose a speed that is 30% lower than pref (nat optimum)
% assume preferred speed is 1.25 and preferred frequency is 110 bpm
speed30p = 1.25*0.70;
s1Speed = cell2mat(ss(1));
[~,i30p] = min(abs(speed30p-s1Speed));
speedHold = s1Speed(i30p);
% convert to % from pref
speedHold = ((s1Speed(i30p)-Speedmin)./Speedmin).*100;

% find the frequency at which to constrain them
freq30p = cell2mat(ff2(1));
% convert to bpm as the cost landscape is in these units
freqHold = freq30p(i30p)*60;
% convert to % from pref
freqHold = ((freq30p(i30p)*60-Freqmin)./Freqmin).*100;

%% Define characteristics of learning agent
% Below is Jess' simple RL algorithm implemented in a multidimensional learning space

% execution noise (action variabilty)
exec_noise=1; % spm

% measurement noise (cost variabilty)
meas_noise=0.02; 

% weighting of new measurements (learning rate)
alpha=0.5; 

% for now, we will only look at spontaneous initiators
int_opt=1; % this is simply set to always be optimizing

%% Define characteristics of protocol
% for now, we will not give the agent explicit experience with the cost landscape
% total amount of steps
steps = 600*2;

% when to switch from hold to release
tHold = 600*1;

%% Define characteristics of analysis

% number of repeats (experiments or subjects)
repeats=10;  % use for quick partial simulation
% repeats=1000;  % use for full simulation

% pre-allocate variables
steps_all = nan(steps,1);
reward_all = nan(steps,1);
action_all = nan(steps,n);
reward_all_all = nan(repeats,steps);
action_all_all = nan(repeats,n,steps);

%% Loop through subjects/experiments
for r=1:repeats
    r
    
    % the following is re-set for new subject
    Qa_est=Qa_nat; %randn(size(Qa_nat(:))); % Qa_nat; % inital estimate of cost in natural (will be updated)
    
    % evaluate for each step
    for s=1:steps
        % one dimensional landscape for frequency hold
        if s == 1
            Frange = find(round(Frequency) == (round(freqHold)));
            Qa_nat = Qa_nat(Frange);
            Qa_est = Qa_nat;
            % make units in % from preferred (or optimal)
            Speed = ((S(Frange) - Speedmin)./Speedmin).*100;
            Frequency = ((F(Frange) - Freqmin)./Freqmin).*100;
            n = 1;
            % for dealing with edges
            maxlength = length(Speed);
            
        % two dimensional landscape for release
        elseif s == tHold
            Qa_nat = E(:);
            Qa_est = Qa_nat;
            Speed = ((S - Speedmin)./Speedmin).*100;
            Frequency = ((F - Freqmin)./Freqmin).*100;
            n = 2;
            % for dealing with edges
            maxlength = size(E);
            
        end
            
        % actions
        % Qa_est contains every possible combination of actions
        [~,ind] = min(Qa_est(:));
        
        % dynamically sized variable to store actions
        a = cell([1 n]);
        
        % convert to individual actions in each dimension
        [a{:}]=ind2sub(size(E),ind);
        a = cell2mat(a);
        
        % frequency (rows), speed (columns)
        % add execution noise to step frequency and width
        anew = a + exec_noise.*randn(1,length(a));
        anew = round(anew);
        
        for i = 1:n
            if anew(i) < 1
                anew(i) = 1;
            elseif anew(i) > maxlength(i)
                anew(i) = maxlength(i);
            end
        end
        
        if n == 1
            action = [Frequency(anew) + exec_noise.*randn(1,length(a)) Speed(anew)];
            % for plotting
            [~,minActual] = min(Qa_nat(:));
            OptimalandHold(s,:) = [freqHold Speed(minActual)];
        else
            action = [Frequency(anew(1),1) Speed(1,anew(2))];
            % for plotting
            [~,minActual] = min(Qa_nat(:));
            OptimalandHold(s,:) = [Frequency(minActual) Speed(minActual)];
        end
        
        % convert back to single index for Q
        anew=num2cell(anew);
        actionIndx = sub2ind(size(Speed),anew{:});
        
        % solve for the reward for the action
        reward = Qa_nat(actionIndx) + meas_noise.*randn;

        % Update estimate of cost landscape based on action
        % update the estimate of this point in the cost landscape
        Qa_est(actionIndx)=Qa_est(actionIndx) + alpha*(reward-Qa_est(actionIndx));
        
        % log data for each step
        steps_all(s)=s;
        reward_all(s)=reward;
        action_all(s,:)=action;
%         Qa_est_all(s,:)= Qa_est;
    end
    
    % log data across repeats
    reward_all_all(r,:)=reward_all';
    action_all_all(r,:,:)=action_all';
%     Qa_est_all_all(:,:,r)=Qa_est_all;
end

% calculating means
action_mean = squeeze(mean(action_all_all))';

%% saving
if 0
    filename = strcat('DimensionTimeConstant',num2str(n),'.mat');
    save(filename,'steps_all','steps','num','action_mean','subsNat','subsNew')
end

%% calculate time constants
% action_mean = mean(action_mean);
% 
% % fit with exponenetial
% ft = fittype(@(a,b,x) a*(1-exp(-x/b)));
% curve = fit(steps_all,(action_mean)',ft);
% expfit = curve.a.*(1-exp(-steps_all./curve.b));
% % get rise time
% S = stepinfo(expfit,steps_all,expfit(i,end),'RiseTimeLimits',[0,0.95]);
% riseTime = S.RiseTime;
% 
% riseTimeAvg = mean(riseTime);
% disp(strcat('riseTime=',num2str(riseTimeAvg)))

%% Plot the outcome, averaged across subjects
j = figure(2);
subplot(2,1,1)
hold on
plot(steps_all,action_mean(:,2),'b')
plot(steps_all,OptimalandHold(:,2),'k')
plot([tHold tHold],[-20 10],'r')
ylim([-20 10])
title('Speed')

subplot(2,1,2)
hold on
plot(steps_all,action_mean(:,1),'b')
plot(steps_all,OptimalandHold(:,1),'k')
plot([tHold tHold],[-20 10],'r')
ylim([-20 10])
title('Frequency')

if 0
    filename = strcat('Dimension',num2str(n),'.png');
    saveas(h,filename)
end