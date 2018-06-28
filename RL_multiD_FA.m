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
[b,natOptS,natOptF] = costLandscapes(n);

%% Define characteristics of learning agent
% Below is Jess' simple RL algorithm implemented in a multidimensional learning space

% execution noise (action variabilty)
exec_noise=1; % spm

% measurement noise (cost variabilty)
meas_noise=0.02;

% forgetting factor for function approximation
% The smaller lambda  is, the smaller is the contribution of previous samples 
% This makes it more sensitive to recent samples
lambda = 0.95;

%% Define characteristics of protocol
% for now, we will not give the agent explicit experience with the cost landscape
steps = 600*2; % corresponding to 300s hold, 300s release

% load pref speed when freq constrained overground
load('v-f_data')
% ff2 corresponds to the prescribed step frequencies during the overground constrained 
% experiments and ss corresponds to the corresponding steady-state speeds.

% choose a speed that is 30% lower than pref (nat optimum)
speed30p = natOptS*0.70;
s1Speed = cell2mat(ss(1));
[~,i30p] = min(abs(speed30p-s1Speed));
speedHold = s1Speed(i30p);

% find the frequency at which to constrain them
freq30p = cell2mat(ff2(1));
% convert to bpm as the cost landscape is in these units
freqHold = freq30p(i30p)*60;

%% Define characteristics of analysis
% number of repeats (experiments or subjects)
repeats=10;  % use for quick partial simulation
% repeats=1000;  % use for full simulation

% pre-allocate variables
gamma = nan(size(b,1),n);
rewardNat = nan(steps,n);
actionNat = nan(steps,n);
bNat_all = nan(size(b,1)-1,n,steps);
stepsNat_all = nan(steps,1);
rewardNat_all = nan(steps,n,repeats);
actionNat_all = nan(steps,n,repeats);

%% Loop through subjects/experiments
for r=1:repeats
    r

    %% Learn natural cost landscape
    % the following is parameter guess re-set for new subject

    % initial parameter guess for all dimensions
    bNat = randn(2,n); % random guess  
    R = 1000.*eye(2*n); % This is used in RLS and it will converge over time as it learns. 

    for sNat=1:steps
        % choose action given predicted cost
        a = -bNat(1,:)./(2*bNat(2,:));

        % add variability to action
        if sNat < 600
            action(sNat,:) = [speedHold freqHold] + exec_noise*randn(1,length(a));
        else
            action(sNat,:) = a + exec_noise*randn(1,length(a)); % a random new action centered about the estimated optimal value
        end
        % get 1 reward
        reward = sum((b(2,:).*action(sNat,:) + b(3,:).*(action(sNat,:)).^2),2) + (meas_noise*randn); % the resulting cost with some noise

        % RLS
        % define variables as x, y, theta for RLS notation
        theta = bNat;

        % predicted value (need to better understand why this is a vector of partial derivatives)
        k = 1;
        for i = 1:n
            % grouping partial derivatives for all dimensions in one vector
            x(k:k+1,1) = [action(sNat,i); (action(sNat,i)).^2]; % here's the gradient at the new action
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

        % log data for each step
        rewardNat(sNat,:) = reward;
        actionNat(sNat,:) = action(sNat,:);
        stepsNat_all(sNat)=sNat;
        bNat_all(:,:,sNat) = bNat;
    end

    % log data across repeats
    rewardNat_all(:,:,r)=rewardNat;
    actionNat_all(:,:,r)=actionNat;

    bNat_all_all(:,:,:,r)=bNat_all;
end

% calculating means
actionNat_mean = mean(actionNat_all,3);
bNat_mean = mean(bNat_all_all,4);

%% saving
if 0
    filename = strcat('overground_spsf',num2str(n),'.mat');
    save(filename,'stepsNat_all','stepsNew_all','actionNat_mean','actionNew_mean')
end

%% plotting
j = figure(2);
subplot(2,1,1)
hold on
plot(stepsNat_all,actionNat_mean(:,1))
plot([stepsNat_all(1) stepsNat_all(end)],[natOptS natOptS],'-k')
ylim([0 5])
title('Speed')

subplot(2,1,2)
hold on
plot(stepsNat_all,actionNat_mean(:,2))
plot([stepsNat_all(1) stepsNat_all(end)],[natOptF natOptF],'-k')
ylim([0 150])
title('Frequency')