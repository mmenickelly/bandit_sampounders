% script to test sam pounders on either a generalized rosenbrock or cube objective.  

global max_t

%% To run this test, you must ensure that the following are on your MATLAB
% path: 
% /path/to/IBCDFO/minq/m/minq5
% /path/to/IBCDFO/pounders/m
% /path/to/IBCDFO/pounders/m/general_h_funs

% With that said, this is hard-coded for MY filesystem, you must edit as 
% appropriate for however you have IBCDFO installed. 
addpath('~/IBCDFO/minq/m/minq5')
addpath('~/IBCDFO/pounders/m')
addpath('~/IBCDFO/pounders/m/general_h_funs')
addpath('../'); % don't edit this one, this is just getting sam_pounders on the path. 

% fixed across all experiments: 
problem = 'damped_oscillator';
macro_seed = 88;

n = 4; % 4 parameters, fixed. 
max_t = 20; % how long of a time horizon to use
m = 40; % change number of timepoints to whatever makes sense for experiments. 
t = linspace(0, max_t, m);

% get ready to generate some noise: 
rng(macro_seed);
sigma = 1e-2;

% define ground truth
truth = zeros(1, n);
truth(1) = 1.5; %A - amplitude
truth(2) = 0.2; %gamma - damping coefficient
truth(3) = 2.0; %omega - angular frequency
truth(4) = pi/2; %phi - phase offset
experimental_values = damped_oscillator(truth, 1:m, t);

% add some gaussian noise: 
experimental_values = experimental_values + truth(1) * sigma * randn(m, 1); 

% to see what oscillator looks like. 
%figure; plot(t, experimental_values);

X0 = truth + 0.1 * ones(1, n); 
Low = zeros(1, n);
Upp = Inf*ones(1, n);

npmax = 2*n + 1;
g_tol = 1e-4;
delta_0 = 0.1;
nfs = 1;
xkin = 1;
printf = 1;
spsolver = 2;

Options = [];
Options.printf = 1;

num_seeds = 30;
H = NaN * ones(50 * m * n, 4, num_seeds);
effort = H; 

for seed = 1:num_seeds
    macro_seed = seed;
    
    nf_max = 50 * n;
    
    % run pounders as a baseline 
    fun = @(x)damped_oscillator_residual(x, 1:m, t, experimental_values);
    
    % define Prior data
    X_init = [X0; repmat(X0, n, 1) + delta_0 * eye(n)];
    F_init = zeros(n+1, m);
    for i = 1:(n+1)
        F_init(i, :) = fun(X_init(i, :));
    end
    nfs = n + 1;
    
    Prior.X_init = X_init; Prior.F_init = F_init; Prior.nfs = n + 1; Prior.xk_in = 1;
    %Prior = [];
    
    [~,~,hF_pounders] = pounders(fun, X_init, n, nf_max, g_tol, delta_0, m, Low, Upp, Prior, Options);
    effort_pounders = linspace(m, m * length(hF_pounders), length(hF_pounders));
    H(1:length(hF_pounders), 1, seed) = hF_pounders;
    effort(1:length(hF_pounders), 1, seed) = effort_pounders;
    
    % sam pounders
    fun = @(x, Set)damped_oscillator_residual(x, Set, t, experimental_values);
    
    nf_max = 50*m*n;
    batch_size = 4;
    
    % expert_array = {@damped_oscillator_custom_policy}; 
    expert_array = {@damped_oscillator_argo_polcy};

    [X_inc_gemini, effort_gemini, models] = sam_pounders_paper(fun, X_init, n, nf_max, g_tol, delta_0, m, Low, Upp, batch_size, expert_array, Prior, Options, []);
    nX = size(X_inc_gemini, 1);
    hF_gemini = zeros(1, nX);
    for j = 1:nX
        hF_gemini(j) = hfun(fun(X_inc_gemini(j, :), 1:m));
    end
    H(1:length(hF_gemini), 2, seed) = hF_gemini;
    effort(1:length(hF_gemini), 2, seed) = effort_gemini;

    
    expert_array = {@uniform_policy}; 
    [X_inc_uni, effort_uni, models] = sam_pounders_paper(fun, X_init, n, nf_max, g_tol, delta_0, m, Low, Upp, batch_size, expert_array, Prior, Options, []);
    nX = size(X_inc_uni, 1);
    hF_uni = zeros(1, nX);
    for j = 1:nX
        hF_uni(j) = hfun(fun(X_inc_uni(j, :), 1:m));
    end
    H(1:length(hF_uni), 3, seed) = hF_uni;
    effort(1:length(hF_uni), 3, seed) = effort_uni;
    
    
    expert_array = {@lipschitz_estimate_policy}; 
    [X_inc_lip, effort_lip, models] = sam_pounders_paper(fun, X_init, n, nf_max, g_tol, delta_0, m, Low, Upp, batch_size, expert_array, Prior, Options, []);
    nX = size(X_inc_lip, 1);
    hF_lip = zeros(1, nX);
    for j = 1:nX
        hF_lip(j) = hfun(fun(X_inc_lip(j, :), 1:m));
    end
    H(1:length(hF_lip), 4, seed) = hF_lip;
    effort(1:length(hF_lip), 4, seed) = effort_lip;

    save('oscillator_results.mat', 'H', 'effort', '-mat');
    if seed == 1
        figure;
    end
    plot(effort_pounders, hF_pounders, 'Color', 'r');
    hold on
    plot(effort_uni, hF_uni, 'Color', 'y')
    plot(effort_lip, hF_lip, 'Color', 'b')
    plot(effort_gemini, hF_gemini, 'Color', 'g')
end



% num_iters = length(models(1).critical_iters);
% nfcount = zeros(m, num_iters);
% for j = 1:m
%     nfcount(j, 1:length(models(j).critical_iters)) = models(j).critical_iters;
%     nfcount(j, 1:length(models(j).trial_iters)) = nfcount(j, 1:length(models(j).trial_iters)) + models(j).trial_iters;
%     nfcount(j, 1:length(models(j).improve_iters)) = nfcount(j, 1:length(models(j).improve_iters)) + models(j).improve_iters;
%     nfcount(j, 1:length(models(j).update_iters)) = nfcount(j, 1:length(models(j).update_iters)) + models(j).update_iters;
% end
% % the logic of the algorithm allows for iterations that don't have ANY
% % evaluations. these are uninteresting in some sense, so we'll just remove
% % them from this summary figure. 
% nfcount(:,all(nfcount == 0))=[];
% figure;
% %spy(nfcount);
% spyc(nfcount');
