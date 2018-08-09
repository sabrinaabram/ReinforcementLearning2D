% Created 27 June 2018
% The purpose of this is to code Renato's preferred walking speed protocol
% and create a general function for us to insert different ways of
% representing the cost landscape

clear
close all
set(0,'defaultAxesFontSize',14)

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

%% Cost landscapes
% dimensions
n = 2;

% create separate function that defines the cost landscape
[bActual,natOptS,natOptF,Spref,Fpref] = costLandscapes_v2(n);

%% Define characteristics of analysis
% number of repeats (experiments or subjects)
repeats=10;  % use for quick partial simulation
% repeats=1000;  % use for full simulation

% pre-allocate variables
gamma = nan(size(bActual,1),n);
reward = nan(steps,1);
action = nan(steps,n);
bEst_all = nan(size(bActual,1)-1,n,steps);
steps_all = nan(steps,1);
reward_all = nan(steps,1,repeats);
action_all = nan(steps,n,repeats);

%% Loop through subjects/experiments
for r=1:repeats
    r

    %% Learn natural cost landscape
    % the following is parameter guess re-set for new subject

    % initial parameter guess for all dimensions
    bEst = randn(2,n); % random guess  
    R = 1000.*eye(2*n); % This is used in RLS and it will converge over time as it learns. 

    for s=1:steps
        % choose action given predicted cost
        a = -bEst(1,:)./(2*bEst(2,:));

        % add variability to action
        if s < tHold
            Speedaction = a(1);
            action(s,:) = [Speedaction freqHold] + exec_noise*randn(1,n);
        else
            action(s,:) = a + exec_noise*randn(1,length(a)); % a new action centered about the estimated optimal value
        end
        % get 1 reward
        reward = sum((bActual(2,:).*action(s,:) + bActual(3,:).*(action(s,:)).^2),2) + (meas_noise*randn); % the resulting cost with some noise

        % RLS
        % define variables as x, y, theta for RLS notation
        theta = bEst;

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
        bEst = reshape(theta,2,n);

        % log data for each step
        reward(s,:) = reward;
        action(s,:) = action(s,:);
        steps_all(s)=s;
        bEst_all(:,:,s) = bEst;
    end

    % log data across repeats
    reward_all(:,:,r)=reward;
    action_all(:,:,r)=action;

    bEst_all_all(:,:,:,r)=bEst_all;
end

% calculating means
action_mean = mean(action_all,3);
bEst_mean = mean(bEst_all_all,4);

%% saving
if 0
    filename = strcat('overground_spsf',num2str(n),'.mat');
    save(filename,'steps_all','action_mean')
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
ylim([-30 15])
title('Frequency')