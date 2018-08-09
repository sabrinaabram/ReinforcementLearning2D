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
[bActual,natOptS,natOptF,Spref,Fpref] = costLandscapes_v2(n);

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

% learning rate for stochastic gradient descent
alpha = 0.6;

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
% assume preferred speed is 1.25 and preferred frequency is 110 bpm
speed30p = 1.25*0.70;
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

% pre-allocate variables
gamma = nan(size(bActual,1),n);
reward = nan(steps);
action = nan(steps,n);
bNat_all = nan(size(bActual,1)-1,n,steps);
steps_all = nan(steps,1);
reward_all = nan(steps,repeats);
action_all = nan(steps,n,repeats);

%% Loop through subjects/experiments
for r=1:repeats
    r

    %% Learn natural cost landscape
    clearvars x

    %% with stochastic policy gradient descent
    
    % the following is parameter guess re-set for new subject
    % initial parameter guess for all dimensions
    bNat = rand(2,2); %b(2:end,:);
    % choose action given predicted cost
    action = -bNat(1,:)./(2*bNat(2,:)) + exec_noise*randn(1,n);
    % This is used in RLS and it will converge over time as it learns
    R = 1000.*eye(2);
    
    for s=1:steps
        % one dimensional landscape for frequency hold
        if s < tHold
            
            % start gradient descent on second iteration for at least 2 data points
            if s > 1
                % choose minimal cost speed given predicted cost and frequency
                % constraint
                % first columns is speed

                % could change it to s^(-alpha) for the learning rate to
                % decay over time and eliminate oscillations
                a(s,:)=action(s-1,:)-alpha.*(costLandscapesDerivative(action(s-1,:),bNat) + randn);

                % add variability to action
                action(s,:) = [a(s,1) freqHold] + exec_noise*randn(1,n);
                
                % reward at new state
                reward = sum((bActual(2,:).*action(s,:) + bActual(3,:).*(action(s,:)).^2),2) + (meas_noise*randn);
            else
                % reward at state
                reward = sum((bActual(2,:).*action(s,:) + bActual(3,:).*(action(s,:)).^2),2) + (meas_noise*randn); % the resulting cost with some noise
            end

            % define variables as x, y, theta for RLS notation
            % this is 2 (linear and quadratic speed terms) compared to 4 in the 2D landscape
            % speed is the first column, and that is the only landscape we
            % are learning here
            theta = bNat(:,1);

            % predicted value (need to better understand why this is a vector of partial derivatives)
            k = 1;
            for i = 1:n-1 % because of frequency contraint
                % grouping partial derivatives for all dimensions in one vector
                x(k:k+1,1) = [action(s,i); (action(s,i)).^2]; % here's the gradient at the new action
                k = k + 2;
            end

            % observed value
            y = reward;

            % update covariance matrix
            R = (1/lambda)*(R - (R*x*x'*R)/(lambda+x'*R*x));

            % Kalman gain
            K = R*x;
            % prediction error
            e = y-x'*theta(:);

            % recursive update for parameter vector
            theta = theta(:) + K*e;

            % convert back
            % b is the actual value, and this is used for frequency because
            % it is known from everyday walking and it is constrained
            % theta is predicted b value for speed because it is being
            % learned from not being constrained
            bNat = [theta bActual(2:end,2)];
        
        % two dimensional landscape for release
        else
            if s == tHold
                % initial parameter guess for all dimensions
                R = 1000.*eye(2*n); % This is used in RLS and it will converge over time as it learns.
            end
        
            % start gradient descent on second iteration for at least 2 data points
            if s > 1
                % could change it to s^(-alpha) for the learning rate to
                % decay over time and eliminate oscillations
                a(s,:)=action(s-1,:)-alpha.*(costLandscapesDerivative(action(s-1,:),bNat));

                % add variability to action
                action(s,:) = a(s,:) + exec_noise*randn(1,n);
                
                % reward at new state
                reward = sum((bActual(2,:).*action(s,:) + bActual(3,:).*(action(s,:)).^2),2) + (meas_noise*randn);
            else
                % reward at state
                reward = sum((bActual(2,:).*action(s,:) + bActual(3,:).*(action(s,:)).^2),2) + (meas_noise*randn); % the resulting cost with some noise
            end

            % define variables as x, y, theta for RLS notation
            theta = bNat;

            % predicted value (need to better understand why this is a vector of partial derivatives)
            k = 1;
            for i = 1:n
                % grouping partial derivatives for all dimensions in one vector
                x(k:k+1,1) = [action(s,i); (action(s,i)).^2]; % here's the gradient at the new action
                k = k + 2;
            end

            % observed value
            y = reward;

            % update covariance matrix
            R = (1/lambda)*(R - (R*x*x'*R)/(lambda+x'*R*x));

            % Kalman gain
            K = R*x;
            % prediction error
            e = y-x'*theta(:);

            % recursive update for parameter vector
            theta = theta(:) + K*e;

            % convert back
            bNat = reshape(theta,2,n);
        end

        % log data for each step
        reward(s,:) = reward;
        action(s,:) = action(s,:);
        steps_all(s)=s;
        bNat_all(:,:,s) = bNat;
    end

    % log data across repeats
    reward_all(:,r)=reward;
    action_all(:,:,r)=action;

    bNat_all_all(:,:,:,r)=bNat_all;
end

% calculating means
action_mean = mean(action_all,3);
% action_mean2 = mean(action_all2,3);

%% saving
if 0
    filename = strcat('overground_spsf',num2str(n),'.mat');
    save(filename,'steps_all','stepsNew_all','action_mean','actionNew_mean')
end

%% plotting
j = figure(2);
subplot(2,1,1)
hold on
plot(steps_all,action_mean(:,1),'b')
plot([steps_all(1) steps_all(end)],[natOptS natOptS],'-k')
plot([tHold tHold],[-20 20],'r')
ylim([-15 15])
title('Speed')

subplot(2,1,2)
hold on
plot(steps_all,action_mean(:,2),'b')
plot([steps_all(1) steps_all(end)],[natOptF natOptF],'-k')
plot([tHold tHold],[-20 20],'r')
ylim([-15 15])
title('Frequency')