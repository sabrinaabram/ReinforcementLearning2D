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

% degree of each cost landscape polynomial in each dimension
expns = 2;

% dimension is a discretized range from -15 SD to +15 SD. Currently, we are
% discretizing this into 20 bins, but this may be increased or decreased
% depending on the number of dimensions as possible combinations grow
% exponentially with dimensions.
num = -15:15; % uint8?
a = permn(num,n);

% shift to -3 SD for all dimensions to simply things, and since we are
% not thinking about sw/sf here, but arbitrary dimensions
if n == 1
    % natural objective function
    Qa_nat = 10*(a./100).^expns + 1;
    % new objective function
    Qa_new = 2*10*(a./100-3/100).^expns + 1;
else
    % natural objective function
    Qa_nat = sum(10*(a./100).^expns')' + 1;
    % new objective function
    Qa_new = sum(2*10*(a./100-3/100).^expns')' + 1;
end

% clearvars a

% natural cost landscape
% find the energy minimum
[~,indNat] = min(Qa_nat(:));
subsNat = cell([1 n]);  % dynamically sized variable
[subsNat{:}]=ind2sub(repmat(20,1,n),indNat);
subsNat = cell2mat(subsNat);

% new cost landscape
% find the energy minimum
[~,indNew] = min(Qa_new(:));
subsNew = cell([1 n]);  % dynamically sized variable
[subsNew{:}]=ind2sub(repmat(20,1,n),indNew);
subsNew = cell2mat(subsNew);

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
steps = 2000; % number of total steps.
con=ones(steps,1); % number of steps with controller on.

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
    Qa_est=double(Qa_nat);  % inital estimate of cost in natural (will be updated)
    
    % evaluate for each step
    for s=1:steps
    
        % actions
        % Qa_est contains every possible combination of actions
        [~,ind] = min(Qa_est(:));
        
        % dynamically sized variable to store actions
        action = cell([1 n]);
        
        % convert to individual actions in each dimension
        [action{:}]=ind2sub(repmat(length(num),1,n),ind); % repmat(20,1,n) is size of Q in matrix form
        action = cell2mat(action);
        
        % add execution noise to step frequency and width
        anew = action + exec_noise.*randn(1,length(action));
        anew = round(anew);
        
        % deal with edges
        for i = 1:n
            if anew(i) < 1
                anew(i) = 1;
            elseif anew(i) > length(a)
                anew(i) = length(a);
            end
        end
        
        % convert back to single index for Q
        if n > 1
            anew=num2cell(anew);
            actionIndx = sub2ind(repmat(length(num),1,n),anew{:});
        else
            actionIndx = anew;
        end
        
        % solve for the reward for the action
        if con(s)==0 % controller off
            reward = Qa_nat(actionIndx) + meas_noise.*randn;
        else % controller on
            reward = Qa_new(actionIndx) + meas_noise.*randn;
        end

        % Update estimate of cost landscape based on action
        % update the estimate of this point in the cost landscape
        Qa_est(actionIndx)=Qa_est(actionIndx) + alpha*(reward-Qa_est(actionIndx));
        
        % log data for each step
        steps_all(s)=s;
        reward_all(s)=reward;
        if n > 1
            action_all(s,:)=num(cell2mat(anew));
        else
            action_all(s,:)=num(anew);
        end
%         Qa_est_all(s,:)= Qa_est;
    end
    
    % log data across repeats
    reward_all_all(r,:)=reward_all';
    action_all_all(r,:,:)=action_all';
%     Qa_est_all_all(:,:,r)=Qa_est_all;
end

% calculating means
action_mean = squeeze(mean(action_all_all));
if n == 1
    action_mean = action_mean';
end

%% saving
if 0
    filename = strcat('DimensionTimeConstant',num2str(n),'.mat');
    save(filename,'steps_all','steps','num','action_mean','subsNat','subsNew')
end

%% calculate time constants
action_mean = mean(action_mean);

% fit with exponenetial
ft = fittype(@(a,b,x) a*(1-exp(-x/b)));
curve = fit(steps_all,(action_mean)',ft);
expfit = curve.a.*(1-exp(-steps_all./curve.b));
% get rise time
S = stepinfo(expfit,steps_all,expfit(i,end),'RiseTimeLimits',[0,0.95]);
riseTime = S.RiseTime;

riseTimeAvg = mean(riseTime);
disp(strcat('riseTime=',num2str(riseTimeAvg)))

%% Plot the outcome, averaged across subjects
h = figure(2);
hold on
plot(steps_all,action_mean,'b')
plot(steps_all,expfit,'r')
plot([0,steps],[0 0],'--k', 'LineWidth',1)
plot([0,steps],[3 3],'--r','LineWidth',1)
ylim([-10 10])
xlim([0 steps])
title(strcat('Dimension',num2str(i)))
xlabel('Steps')
ylabel('Action (sd from pref)')

if 0
    filename = strcat('Dimension',num2str(n),'.png');
    saveas(h,filename)
end