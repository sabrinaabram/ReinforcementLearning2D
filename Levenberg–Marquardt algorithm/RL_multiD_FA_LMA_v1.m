% Created 27 June 2018
% The purpose of this is to code Renato's preferred walking speed protocol
% and create a general function for us to insert different ways of
% representing the cost landscape

clear
close all
set(0,'defaultAxesFontSize',14)

%% Cost landscapes
% dimensions
n = 2;

% create separate function that defines the cost landscape
[bActual,Spref,Fpref] = costLandscapes_v3(n);

%% Define characteristics of learning agent
% Below is Jess' simple RL algorithm implemented in a multidimensional learning space

% execution noise (action variabilty)
exec_noise=1; % think about this for future speed and freq because absolute value

% measurement noise (cost variabilty)
meas_noise=0.02;

% forgetting factor for function approximation
% The smaller lambda  is, the smaller is the contribution of previous samples 
% This makes it more sensitive to recent samples
lambda = 0.99;

%% Define characteristics of protocol
% total amount of steps
steps = 600*2;

% when to switch from hold to release
tHold = 600*1;

% choose a speed that is 30% lower than pref for the hold (nat optimum) for
% the hold
freqHold = -30;

%% Define characteristics of analysis
% number of repeats (experiments or subjects)
repeats=10;  % use for quick partial simulation
% repeats=1000;  % use for full simulation

% % pre-allocate variables
% gamma = nan(size(bActual,1),n);
% reward_all = nan(steps,1);
% action_all = nan(steps,n);
% bEst_all = nan(size(bActual,1)-1,n,steps);
% steps_all = nan(steps,1);
% reward_all_all = nan(steps,repeats);
% action_all_all = nan(steps,n,repeats);

%% Loop through subjects/experiments
for r=1:repeats
    r
    clearvars x

    %% with stochastic policy gradient descent
    
    for s=1:steps
        if s == 1
            % initial parameter guess
%             bEst = bActual; %randn(length(bActual),1); %bActual; % random guess  
            bEst = [0.5958    2.0533   -1.5293    0.0227    0.0953]; % random guess, but consistent for repeats
            % choose action given predicted cost
            actionCurr = [0 0];
            % This is used in RLS and it will converge over time as it learns.
            R = 1000.*eye(length(bEst)); 
            % LMA damping constant
            alpha = 1; % 1 = gradient descent, 0 = gauss-newton
            i = 1;
        elseif s == tHold
            % This is used in RLS and it will converge over time as it learns.
            R = 1000.*eye(length(bEst));
            % reset LMA damping constant
            alpha = 1;
            i = 1;
        end
        
        % for comparison purposes
        aOpt = evalOptimum(bActual,s,tHold,freqHold);
        
        % gradient
        gradient = costLandscapesDerivative(actionCurr,bEst);
        
        % hessian
        hessian = costLandscapeHessian(bEst);

        % LMA modified hessian
        h = hessian + alpha*[1;1];
        
        % if lambda is high = gradient descent, low = gauss-newton
        deltax = -gradient./h;
        
        % gradient descent learning
        % this is the gradient wrt the action
        % s^(-alpha) for the learning rate to decay over time and eliminate oscillations
        action = actionCurr + deltax';
        if s < tHold
            action(2) = freqHold; % frequency hold with variability
        end
        
        % add variability to action
        action = action + exec_noise*randn(1,length(action)); % a new action centered about the estimated optimal value
        
        % get actual reward after gradient descent
        reward = bActual(1)*action(1) + bActual(2)*action(2) + bActual(3)*(action(1))^2 + ...
               bActual(4)*(action(1))*(action(2)) + bActual(5)*(action(2))^2 + meas_noise.*randn;
        
        % RLS
        % define variables as x, y, theta for RLS notation
        theta = bEst;

        % vector of partial derivatives
        k = 1;
        x = [action(1) action(2) (action(1))^2 ...
            (action(1))*(action(2)) (action(2))^2]';
        
        % observed value
        y = reward;

        % update covariance matrix
        R = (1/lambda)*(R - (R*x*x'*R)/(lambda+x'*R*x));

        % Kalman gain
        K = R*x;
        % prediction error
        e = y-x'*theta(:);

        % recursive update for parameter vector
        bEst = theta(:) + K*e;

        % log data for each step
        reward_all(s,:) = reward;
        action_all(s,:) = action;
        actionOpt_all(s,:) = aOpt;
        steps_all(s)=s;
        bEst_all(s,:) = bEst';
        
        actionCurr = action;
        i = i + 1;
        alpha = alpha*.9;
    end
        
    % log data across repeats
    reward_all_all(:,:,r)=reward_all;
    action_all_all(:,:,r)=action_all;
    actionOpt_all_all(:,:,r)=actionOpt_all;
    bEst_all_all(:,:,:,r)=bEst_all;
end

% calculating means
% calculating means
action_mean = mean(action_all_all,3);
actionOpt_mean = mean(actionOpt_all_all,3);

%% saving
if 0
    filename = strcat('overground_spsf',num2str(n),'.mat');
    save(filename,'steps_all','stepsNew_all','action_mean','actionNew_mean')
end

%% calculate time constants

% 1st speed adaptation
steps_speed1 = steps_all(1:tHold-1);
ft = fittype(@(a,b,x) a*(1-exp(-x/b)));
curve = fit(steps_speed1',action_mean(1:tHold-1,1),ft);
expfit_speed1 = curve.a.*(1-exp(-steps_speed1./curve.b));
% get rise time
S = stepinfo(expfit_speed1,steps_speed1,expfit_speed1(end),'RiseTimeLimits',[0,0.95]);
riseTime_speed1 = S.RiseTime

% 2nd speed adaptation
steps_speed2 = steps_all(tHold:end);
curve = fit(steps_speed2',action_mean(tHold:end,1),'exp2');
expfit_speed2 = curve.a.*exp(curve.b.*steps_speed2) + curve.c.*exp(curve.d.*steps_speed2);
% get rise time
S = stepinfo(expfit_speed2,steps_speed2,expfit_speed2(end),'RiseTimeLimits',[0,0.95]);
riseTime_speed2 = S.RiseTime

% frequency adaptation
steps_frequency = steps_all(tHold:end);
curve = fit(steps_frequency',action_mean(tHold:end,2),'exp2');
expfit_frequency = curve.a.*exp(curve.b.*steps_frequency) + curve.c.*exp(curve.d.*steps_frequency);
% get rise time
S = stepinfo(expfit_frequency,steps_frequency,expfit_frequency(end),'RiseTimeLimits',[0,0.95]);
riseTime_frequency = S.RiseTime

%% Plot the outcome, averaged across subjects

j = figure(2);
subplot(2,1,1)
hold on
plot(steps_all,action_mean(:,1),'b')
plot(steps_all, actionOpt_mean(:,1),'-k')
plot(steps_speed1,expfit_speed1,'r','LineWidth',2)
plot(steps_speed2,expfit_speed2,'r','LineWidth',2)
ylim([-50 10])
ylabel('Speed')
title('Overground experiment with frequency hold and release')

subplot(2,1,2)
hold on
plot(steps_all,action_mean(:,2),'b')
plot(steps_all, actionOpt_mean(:,2),'-k')
plot(steps_frequency,expfit_frequency,'r','LineWidth',2)
ylim([-50 10])
ylabel('Frequency')
xlabel('Steps')