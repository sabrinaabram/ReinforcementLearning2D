% Created 27 July 2018
% The purpose of this is to code Renato's preferred walking speed protocol
% and use gradient descent to learn value function

clear
close all
set(0,'defaultAxesFontSize',14)

%% Cost landscapes
% dimensions
n = 2;

% create separate function that defines the cost landscape
addpath /Users/sabrinaabram/Desktop/ReinforcementLearning_multiD
[bActual,Spref,Fpref] = costLandscapes_v3(n);

%% Define characteristics of learning agent
% Below is Jess' simple RL algorithm implemented in a multidimensional learning space

% execution noise (action variabilty)
exec_noise=1; % think about this for future speed and freq because absolute value

% measurement noise (cost variabilty)
meas_noise=0.02;

% learning rate for gradient descent
alpha = 1e-9;

%% Define characteristics of protocol
% total amount of steps
steps = 600*2;

% when to switch from hold to release
tHold = 600*1;

% load pref speed when freq constrained overground
load('v-f_data')
% ff2 corresponds to the prescribed step frequencies during the overground constrained 
% experiments and ss corresponds to the corresponding steady-state speeds.

% choose a speed that is 30% lower than pref (nat optimum)
speed30p = Spref*.70;
s1Speed = cell2mat(ss(1));
[~,i30p] = min(abs(speed30p-s1Speed));
speedHold = s1Speed(i30p);
% convert to % from pref
speedHold = ((s1Speed(i30p)-Spref)./Spref).*100;

% find the frequency at which to constrain them
freq30p = cell2mat(ff2(1));
% convert to bpm as the cost landscape is in these units
freqHold = freq30p(i30p)*60;
% convert to % from pref
freqHold = ((freq30p(i30p)*60-Fpref)./Fpref).*100;

%% Define characteristics of analysis
% number of repeats (experiments or subjects)
repeats=10;  % use for quick partial simulation
% repeats=1000;  % use for full simulation

% % pre-allocate variables
% gamma = nan(size(bActual,1),n);
% rewardNat = nan(steps,n);
% actionNat = nan(steps,n);
% bEst_all = nan(size(bActual,1)-1,n,steps);
% steps_all = nan(steps,1);
% rewardNat_all = nan(steps,n,repeats);
% actionNat_all = nan(steps,n,repeats);

%% Loop through subjects/experiments
% gradient descent
for r=1:repeats
    r
    
    % the following is parameter guess re-set for new subject
    % initial parameter guess for all dimensions
    bEst = bActual*randn; %randn(length(bActual),1); %bActual;
    
    for s=1:steps
        % in evalOptimum, we have 1D for s<tHold and 2D for s>=tHold
        a = evalOptimum(bEst, s, tHold, freqHold);
        action = a + exec_noise*randn(1,length(a));
        aOpt = evalOptimum(bActual, s, tHold, freqHold);

        rewardActual = bActual(1)*action(1) + bActual(2)*action(2) + bActual(3)*(action(1))^2 + ...
           bActual(4)*(action(1))*(action(2)) + bActual(5)*(action(2))^2 + meas_noise.*randn;

        rewardEst = bEst(1)*action(1) + bEst(2)*action(2) + bEst(3)*(action(1))^2 + ...
           bEst(4)*(action(1))*(action(2)) + bEst(5)*(action(2))^2 + meas_noise.*randn;
       
        % gradient descent learning value function parameters
        % this is the gradient wrt the 4 estimated function parameters
        gradient = costLandscapesDerivative_b(action,rewardActual,rewardEst);
        % s^(-alpha) for the learning rate to decay over time and eliminate oscillations
        bEst = bEst - alpha.*sign(bEst).*abs(gradient);

        % log data for each step
        reward_all(s,:) = rewardEst;
        action_all(s,:) = action;
        actionOpt_all(s,:) = aOpt;
        steps_all(s)=s;
        bEst_all(s,:) = bEst';
        
        error(s,:) = (bEst-bActual)';
        grad(s,:) = gradient';
    end

    % log data across repeats
    reward_all_all(:,:,r)=reward_all;
    action_all_all(:,:,r)=action_all;
    actionOpt_all_all(:,:,r)=actionOpt_all;
    bEst_all_all(:,:,:,r)=bEst_all;
end

% calculating means
action_mean = mean(action_all_all,3);
actionOpt_mean = mean(actionOpt_all_all,3);

%% saving
if 0
    filename = strcat('overground_spsf',num2str(n),'.mat');
    save(filename,'steps_all','stepsNew_all','actionNat_mean','actionNew_mean')
end

%% plotting
j = figure(2);
subplot(2,1,1)
hold on
plot(steps_all,action_mean(:,1),'b')
plot(steps_all, actionOpt_mean(:,1),'-k')
plot([tHold tHold],[-30 15],'r')
ylim([-20 10])
title('Speed')

subplot(2,1,2)
hold on
plot(steps_all,action_mean(:,2),'b')
plot(steps_all, actionOpt_mean(:,2),'-k')
plot([tHold tHold],[-30 15],'r')
ylim([-20 10])
title('Frequency')