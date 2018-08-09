% Created 22 March 2018
% The purpose of this code is to extend Jess' 1D RL to multiple dimensions,
% and then test how the adaptation time constant depends on the number of
% dimensions

clear
close all
set(0,'defaultAxesFontSize',14)

%% Cost landscapes
% dimensions (can generalize to any number)
n = 1; % speed is constrained, frequency is freely selected

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

% before normalizing, find indexes of speeds for perturbations
S1 = find(S(1,:)==1);
S2 = find(S(1,:)==1.25);

S = ((S - Speedmin)./Speedmin).*100;
F = ((F - Freqmin)./Freqmin).*100;

%% Define characteristics of learning agent
% Below is Jess' simple RL algorithm implemented in a multidimensional learning space

% execution noise (action variabilty)
exec_noise=1; % spm

% measurement noise (cost variabilty)
meas_noise=0.02; 

% weighting of new measurements (learning rate)
alpha=0.8;

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

% % pre-allocate variables
% steps_all = nan(steps,1);
% reward_all = nan(steps,1);
% action_all = nan(steps,n);
% reward_all_all = nan(repeats,steps);
% action_all_all = nan(repeats,n,steps);

%% Loop through subjects/experiments
for r=1:repeats
    r
    
    % evaluate for each step
    for s=1:steps
        % one dimensional landscape for frequency hold
        if s == 1
            Qa_nat = E(:,S1);
        % hypothesis 1
        % Ns considers everything else that it doesnt know to have a large cost
        % what it knows is that pref (0,0) is optimal from years of walking
%             Qa_est = 100.*ones(size(Qa_nat));
%             initSmin = find(S(1,:)==0);
%             Qa_est(initSmin) = Qa_nat(initSmin);
%             Qa_est = Qa_est(:);
        % hypothesis 2
        % NS has representation of landscape at preferred
            initSmin = find(S(1,:)==0);
            Qa_est = E(:,initSmin);
        % hypothesis 3
        % NS abandons all prior knowledge
%             Qa_est = randn(length(Qa_nat),1);
            
            % make units in % from preferred (or optimal)
            Speed = S(:,S1);
            Frequency = F(:,S1);
            n = 1;
            % for dealing with edges
            maxlength = length(Speed);
            
        % two dimensional landscape for release
        elseif s == tHold
            Qa_nat = E(:,S2);
        % hypothesis 1
        % Ns considers everything else that it doesnt know to have a large cost
            Qa_est = 100.*ones(size(Qa_est));
            Qa_est(a) = Qa_nat(a);
        % hypothesis 2
        % NS has representation of landscape at preferred
%             initSmin = find(S(1,:)==0);
%             Qa_est = E(:,initSmin);
%             Qa_est = Qa_est(:);
        % hypothesis 3
        % NS abandons all prior knowledge
%             Qa_est = randn(length(Qa_nat),1);
            
            Speed = S(:,S2);
            Frequency = F(:,S2);
            n = 1;
            % for dealing with edges
            maxlength = length(Speed);
            
        end
        
        % actions
        % Qa_est contains every possible combination of actions
        [~,a] = min(Qa_est);

        % frequency (rows), speed (columns)
        % add execution noise to step frequency and width
        anew = a + exec_noise.*randn;
        anew = round(anew);

        if anew < 1
            anew = 1;
        elseif anew > maxlength
            anew = maxlength;
        end
        
        action = [Frequency(anew) Speed(anew) + exec_noise.*randn];
        % for plotting
        [~,minActual] = min(Qa_nat(:));
        % dynamically sized variable to store actions
        aOpt = cell([1 n]);
        % convert to individual actions in each dimension
        [aOpt{:}]=ind2sub(size(E),minActual);
        aOpt = cell2mat(aOpt);
        OptimalandHold(s,:) = [Frequency(aOpt(1)) Speed(aOpt(1))];
        
        % solve for the reward for the action
        reward = Qa_nat(anew) + meas_noise.*randn;

        % Update estimate of cost landscape based on action
        % update the estimate of this point in the cost landscape
        Qa_est(anew)=Qa_est(anew) + alpha*(reward-Qa_est(anew));
        
        % log data for each step
        steps_all(s)=s;
        reward_all(s)=reward;
        action_all(s,:)=action;
    end
    
    % log data across repeats
    reward_all_all(r,:)=reward_all';
    action_all_all(r,:,:)=action_all';
end

% calculating means
action_mean = squeeze(mean(action_all_all))';

%% saving
if 0
    filename = strcat('DimensionTimeConstant',num2str(n),'.mat');
    save(filename,'steps_all','steps','num','action_mean','subsNat','subsNew')
end

%% calculate time constants
action_mean=action_mean';

% 1st frequency adaptation
steps_frequency1 = steps_all(1:tHold-1);
ft = fittype(@(a,b,x) a*(1-exp(-x/b)));
curve = fit(steps_frequency1',(action_mean(1,1:tHold-1))',ft);
expfit_frequency1 = curve.a.*(1-exp(-steps_frequency1./curve.b));
% get rise time
S = stepinfo(expfit_frequency1,steps_frequency1,expfit_frequency1(end),'RiseTimeLimits',[0,0.95]);
riseTime_frequency1 = S.RiseTime

% 2nd frequency adaptation
steps_frequency2 = steps_all(tHold:end);
curve = fit(steps_frequency2',(action_mean(1,tHold:end))','exp2');
expfit_frequency2 = curve.a.*exp(curve.b.*steps_frequency2) + curve.c.*exp(curve.d.*steps_frequency2);
% get rise time
S = stepinfo(expfit_frequency2,steps_frequency2,expfit_frequency2(end),'RiseTimeLimits',[0,0.95]);
riseTime_frequency2 = S.RiseTime

%% Plot the outcome, averaged across subjects
action_mean=action_mean';

j = figure(2);
subplot(2,1,1)
hold on
plot(steps_all,action_mean(:,2),'b')
plot(steps_all,OptimalandHold(:,2),'k')
ylim([-50 10])
ylabel('Speed')
title('Treadmill experiment with speed hold and release')

subplot(2,1,2)
hold on
plot(steps_all,action_mean(:,1),'b')
plot(steps_all,OptimalandHold(:,1),'k')
plot(steps_frequency1,expfit_frequency1,'r','LineWidth',2)
plot(steps_frequency2,expfit_frequency2,'r','LineWidth',2)
ylim([-50 10])
ylabel('Frequency')
xlabel('Steps')

if 0
    filename = strcat('Dimension',num2str(n),'.png');
    saveas(h,filename)
end