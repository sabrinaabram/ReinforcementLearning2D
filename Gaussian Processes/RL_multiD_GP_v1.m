% Created 22 March 2018
% The purpose of this code is to extend Jess' 1D RL to multiple dimensions,
% and then test how the adaptation time constant depends on the number of
% dimensions

clear
close all
set(0,'defaultAxesFontSize',14)

%% Cost landscapes

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
n = 2; % speed and freq
subsNat = cell([1 n]);  % dynamically sized variable
[subsNati, subsNatj]=ind2sub(size(E),indNat);

Freqmin = F(subsNati,1);
Speedmin = S(1,subsNatj);

F = ((F - Freqmin)./Freqmin).*100;
F = F(:,(end-length(F(:,1))+1):end);
S = ((S - Speedmin)./Speedmin).*100;
S = S(:,(end-length(F(:,1))+1):end);
E = E(:,(end-length(F(:,1))+1):end);

% redefine with new size
Qa_nat = E(:);

%% initialize Gaussian Process (GP)

% define the number of points for speed and frequency
Stest_hold = (S(1,:))';

% Define the kernel function (radial basis) for covariance matrix
% I think this is the spread or variance for each point to be used in the
% radial basis calculation
param = 0.1;
K_S_ss_hold = kernel(Stest_hold,Stest_hold,param); % initial covariance matrix for speed

% Cholesky decomposition (square root) of the covariance matrix
L_S_hold = chol(K_S_ss_hold + 1e-15*eye(length(K_S_ss_hold)),'lower');

%% initialize Gaussian Process (GP) for the hold

% define the number of points for speed and frequency
Stest = S(:);
Ftest = F(:);

% Define the kernel function (radial basis) for covariance matrix
% I think this is the spread or variance for each point to be used in the
% radial basis calculation
K_S_ss = kernel(Stest,Stest,param); % initial covariance matrix for speed
% K_F_ss = kernel(Ftest,Ftest,param); % initial covariance matrix for frequency

% Cholesky decomposition (square root) of the covariance matrix
L_S = chol(K_S_ss + 1e-15*eye(length(K_S_ss)),'lower');
% L_F = chol(K_F_ss + 1e-15*eye(length(K_F_ss)),'lower');

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

% choose a speed that is 30% lower than pref (nat optimum) for the hold
% first, find the energy minimal frequency for each speed
[~,EminAllspeeds] = min(E');
% second find the energy minimal frequency that corresponds to -30% speed
for i = 1:length(EminAllspeeds)
    FminAllspeeds(i) = F(i,EminAllspeeds(i));
    SminAllspeeds(i) = S(i,EminAllspeeds(i));
end

[~,speedHoldi] = min(abs(SminAllspeeds-(-30)));
speedHold = S(speedHoldi,EminAllspeeds(speedHoldi));

freqHold = F(speedHoldi,EminAllspeeds(speedHoldi));

%% Define characteristics of analysis

% number of repeats (experiments or subjects)
repeats=1;  % use for quick partial simulation
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
            n = 1;
            
            Qa_nat = E(speedHoldi,:);
            Qa_est = E(speedHoldi,:);
            Speed = S(speedHoldi,:);
            Frequency = F(speedHoldi,:);
            
            S_ai = subsNatj; % initialize first training point to be natural minimum
            F_ai = speedHoldi;
            
        % two dimensional landscape for release
        elseif s == tHold
            n = 2;
            
            Qa_nat = E(:);
            Qa_est = E(:);
            maxlength = length(Qa_nat);
            Speed = S(:);
            Frequency = F(:);
        end

        if s < tHold
        % add points to training set at each step
            Strain_x(s,1) = S(1,S_ai); % initialize first training point
            Ftrain_x(s,1) = F(F_ai,1); % initialize second training point
            train_y(s,1) = E(F_ai,S_ai);
        else
        s2 = s-599;
            % add points to training set at each step
            Strain_x(s2,1) = S(1,S_ai); % initialize first training point
            Ftrain_x(s2,1) = F(F_ai,1); % initialize second training point
            train_y(s2,1) = E(F_ai,S_ai);
        end
        
        if n == 1
            % Apply the kernel function to our training points
            K_S = kernel(Strain_x, Strain_x, param);
            L_S = chol(K_S + 0.00005*eye(length(Strain_x)),'lower');

            % Compute the mean at our test points
            K_S_s = kernel(Strain_x, Stest_hold, param);
            Lk_S = mldivide(L_S,K_S_s);
            mu_S = Lk_S'*mldivide(L_S,train_y);

            % Draw samples from the posterior at our test points.
            L_S = chol(K_S_ss_hold + 1e-6*eye(length(K_S_ss_hold)) - Lk_S'*Lk_S);
            f_S_post = mu_S + L_S*normrnd(0,1,[length(Stest_hold),1]);

            % actions
            % Qa_est contains every possible combination of actions
            [~,S_ai] = min(f_S_post);
            S_a = Speed(S_ai);
            a = [freqHold S_a];

            % frequency (rows), speed (columns)
            % add execution noise to step frequency and width
            action = a + exec_noise.*randn(1,length(a));

            % for plotting
            [~,minActual] = min(Qa_nat(:));
            % dynamically sized variable to store actions
            aOpt = cell([1 n]);
            % convert to individual actions in each dimension
            [aOpt{:}]=ind2sub(size(E),minActual);
            aOpt = cell2mat(aOpt);
            OptimalandHold(s,:) = [Frequency(aOpt(1)) Speed(aOpt(1))];
            
            % convert back to single index for Q
            actionIndx = S_ai;
        
        else
            % Apply the kernel function to our training points
            K_S = kernel(Strain_x, Strain_x, param);
%             K_F = kernel(Ftrain_x, Ftrain_x, param);
            L_S = chol(K_S + 0.00005*eye(length(Strain_x)),'lower');
%             L_F = chol(K_F + 0.00005*eye(length(Ftrain_x)),'lower');

            % Compute the mean at our test points
            K_S_s = kernel(Strain_x, Stest, param);
%             K_F_s = kernel(Ftrain_x, Ftest, param);
            Lk_S = mldivide(L_S,K_S_s);
%             Lk_F = mldivide(L_F,K_F_s);
            mu_S = Lk_S'*mldivide(L_S,train_y);
%             mu_F = Lk_F'*mldivide(L_F,train_y);

            % Draw samples from the posterior at our test points.
            L_S = chol(K_S_ss + 1e-6*eye(length(Stest)) - Lk_S'*Lk_S);
%             L_F = chol(K_F_ss + 1e-6*eye(length(Ftest)) - Lk_F'*Lk_F);
            f_S_post = mu_S + L_S*normrnd(0,1,[length(Stest),1]);
%             f_F_post = mu_F + L_F*normrnd(0,1,[length(Ftest),1]);

            % actions
            % Qa_est contains every possible combination of actions
            [~,actionIndx] = min(f_S_post);
            S_a = Speed(actionIndx);
            F_a = Frequency(actionIndx);
            a = [F_a, S_a];

            % frequency (rows), speed (columns)
            % add execution noise to step frequency and width
            action = a + exec_noise.*randn(1,length(a));

            % for plotting
            [~,minActual] = min(Qa_nat(:));
%             % dynamically sized variable to store actions
%             aOpt = cell([1 n]);
%             % convert to individual actions in each dimension
%             [aOpt{:}]=ind2sub(size(E),minActual);
%             aOpt = cell2mat(aOpt);
            OptimalandHold(s,:) = [Frequency(minActual) Speed(minActual)];
            
            % deal with edges
            if actionIndx < 1
                actionIndx = 1;
            elseif actionIndx > maxlength
                actionIndx = maxlength;
            end
        end
        
        if s == tHold-1
            clearvars Strain_x Ftrain_x train_y
        end
        
        % solve for the reward for the action
        reward = Qa_nat(actionIndx) + meas_noise.*randn;

        % Update estimate of cost landscape based on action
        % update the estimate of this point in the cost landscape
        Qa_est(actionIndx)=Qa_est(actionIndx) + alpha*(reward-Qa_est(actionIndx));
        
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

% 1st speed adaptation
steps_speed1 = steps_all(1:tHold-1);
ft = fittype(@(a,b,x) a*(1-exp(-x/b)));
curve = fit(steps_speed1',(action_mean(2,1:tHold-1))',ft);
expfit_speed1 = curve.a.*(1-exp(-steps_speed1./curve.b));
% get rise time
S = stepinfo(expfit_speed1,steps_speed1,expfit_speed1(end),'RiseTimeLimits',[0,0.95]);
riseTime_speed1 = S.RiseTime

% 2nd speed adaptation
steps_speed2 = steps_all(tHold:end);
curve = fit(steps_speed2',(action_mean(2,tHold:end))','exp2');
expfit_speed2 = curve.a.*exp(curve.b.*steps_speed2) + curve.c.*exp(curve.d.*steps_speed2);
% get rise time
S = stepinfo(expfit_speed2,steps_speed2,expfit_speed2(end),'RiseTimeLimits',[0,0.95]);
riseTime_speed2 = S.RiseTime

% frequency adaptation
steps_frequency = steps_all(tHold:end);
curve = fit(steps_frequency',(action_mean(1,tHold:end))','exp2');
expfit_frequency = curve.a.*exp(curve.b.*steps_frequency) + curve.c.*exp(curve.d.*steps_frequency);
% get rise time
S = stepinfo(expfit_frequency,steps_frequency,expfit_frequency(end),'RiseTimeLimits',[0,0.95]);
riseTime_frequency = S.RiseTime

%% Plot the outcome, averaged across subjects
action_mean=action_mean';

j = figure(2);
subplot(2,1,1)
hold on
plot(steps_all,action_mean(:,2),'b')
plot(steps_all,OptimalandHold(:,2),'k')
plot(steps_speed1,expfit_speed1,'r','LineWidth',2)
plot(steps_speed2,expfit_speed2,'r','LineWidth',2)
ylim([-50 10])
ylabel('Speed')
title('Overground experiment with frequency hold and release')

subplot(2,1,2)
hold on
plot(steps_all,action_mean(:,1),'b')
plot(steps_all,OptimalandHold(:,1),'k')
plot(steps_frequency,expfit_frequency,'r','LineWidth',2)
ylim([-50 10])
ylabel('Frequency')
xlabel('Steps')

if 0
    filename = strcat('Dimension',num2str(n),'.png');
    saveas(h,filename)
end

%% kernel equation
function K_ss = kernel(a,b,param)
%         sqdist = a.^2 + (b.^2)' - 2*a*b';
%         K_ss = exp(-0.5*(1/param)*(sqdist));
        K_ss = exp(-0.5*(1/param)*pdist2(a,b,'euclidean').^2);
end