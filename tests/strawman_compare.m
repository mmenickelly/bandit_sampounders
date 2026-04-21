% script to test sam pounders on either a generalized rosenbrock or cube objective.  

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

macro_seed = 8;

n = 4; m = 4; 

alpha = 2^8; % how big the imbalance is. 
%alpha = ones(1, m); alpha(m) = m; 

rng(macro_seed);

Low = -Inf * ones(1, n);
Upp = Inf * ones(1, n); 

%X0 = ones(1, n); 
X0 = zeros(1, n);

npmax = 2*n + 1;
g_tol = sqrt(eps);
delta_0 = 0.1;
nfs = 1;
xkin = 1;
printf = 1;
spsolver = 2;

Options = [];
Options.printf = 1;

%hfun = @(F)sum(F.^2);
hfun = @(F)sum(F);

Options.hfun = @(F)sum(F);
combinemodels = @sum_combine;
Options.combinemodels = combinemodels;

% run strawman first
fun = @(x, Set)strawman_synthetic(x, Set, alpha); 
%fun = @(x, Set)generalized_cube(x, Set, alpha);

nf_max = 20*m*n;
batch_size = 1;

expert_array = {@lipschitz_estimate_policy}; 
%expert_array = {@uniform_policy}; 

%surrogate_array = rbf_surrogates(@(x)damped_oscillator_residual(x,1:m,t, experimental_values), n, m, Low, Upp);
%expert_array = {@(models, batch_size, sample_type, data)surrogate_informed_policy(models, batch_size, sample_type, data, surrogate_array, Low, Upp, spsolver)};

[X_inc_strawman, effort_strawman, models_strawman] = strawman_sam_pounders_paper(fun, X0, n, nf_max, g_tol, delta_0, m, Low, Upp, batch_size, expert_array, [], Options, []);
pause()

%expert_array = {@lipschitz_estimate_policy}; 
expert_array = {@uniform_policy}; 

%expert_array = {@lipschitz_estimate_policy, @uniform_policy}; 

%surrogate_array = rbf_surrogates(@(x)damped_oscillator_residual(x,1:m,t, experimental_values), n, m, Low, Upp);
%expert_array = {@(models, batch_size, sample_type, data)surrogate_informed_policy(models, batch_size, sample_type, data, surrogate_array, Low, Upp, spsolver)};
%expert_array = {@(models, batch_size, sample_type, data)surrogate_informed_policy(models, batch_size, sample_type, data, surrogate_array, Low, Upp, spsolver), @uniform_policy};
%expert_array = {@(models, batch_size, sample_type, data)surrogate_informed_policy(models, batch_size, sample_type, data, surrogate_array, Low, Upp, spsolver), @lipschitz_estimate_policy, @uniform_policy};

[X_inc, effort, models] = strawman_sam_pounders_paper(fun, X0, n, nf_max, g_tol, delta_0, m, Low, Upp, batch_size, expert_array, [], Options, []);

%% POST PROCESSING
FS = 18; 
LW = 4; 

num_iters = nf_max;
nfcount_strawman = zeros(m, num_iters);
for j = 1:m
    nfcount_strawman(j, 1:length(models_strawman(j).critical_iters)) = models_strawman(j).critical_iters;
    nfcount_strawman(j, 1:length(models_strawman(j).trial_iters)) = nfcount_strawman(j, 1:length(models_strawman(j).trial_iters)) + models_strawman(j).trial_iters;
    nfcount_strawman(j, 1:length(models_strawman(j).improve_iters)) = nfcount_strawman(j, 1:length(models_strawman(j).improve_iters)) + models_strawman(j).improve_iters;
    nfcount_strawman(j, 1:length(models_strawman(j).update_iters)) = nfcount_strawman(j, 1:length(models_strawman(j).update_iters)) + models_strawman(j).update_iters;
end
% the logic of the algorithm allows for iterations that don't have ANY
% evaluations. these are uninteresting in some sense, so we'll just remove
% them from this summary figure. 
nfcount_strawman(:,all(nfcount_strawman == 0))=[];

nfcount = zeros(m, num_iters);
for j = 1:m
    nfcount(j, 1:length(models(j).critical_iters)) = models(j).critical_iters;
    nfcount(j, 1:length(models(j).trial_iters)) = nfcount(j, 1:length(models(j).trial_iters)) + models(j).trial_iters;
    nfcount(j, 1:length(models(j).improve_iters)) = nfcount(j, 1:length(models(j).improve_iters)) + models(j).improve_iters;
    nfcount(j, 1:length(models(j).update_iters)) = nfcount(j, 1:length(models(j).update_iters)) + models(j).update_iters;
end
% the logic of the algorithm allows for iterations that don't have ANY
% evaluations. these are uninteresting in some sense, so we'll just remove
% them from this summary figure. 
nfcount(:,all(nfcount == 0))=[];

figure;
t = tiledlayout(2, 2);
nexttile([2 1]);
% plot
% post process
num_evals_strawman = length(effort_strawman);
hF_strawman = zeros(1, num_evals_strawman);
for j = 1:num_evals_strawman
    hF_strawman(j) = hfun(fun(X_inc_strawman(j, :), 1:m));
end
num_evals_uni = length(effort);
hF = zeros(1, num_evals_uni);
for j = 1:num_evals_uni
    hF(j) = hfun(fun(X_inc(j, :), 1:m));
end
semilogy(1:num_evals_strawman, hF_strawman, 'LineWidth', LW, 'LineStyle', '-');
hold on
semilogy(1:num_evals_uni, hF, 'LineWidth', LW, 'LineStyle', '--');
legend('Lipschitz sampling', 'Uniform sampling', 'FontSize', FS)
xlabel('$k$','interpreter','latex', 'FontSize', FS);
ylabel('$f(x^k)$','interpreter','latex', 'FontSize', FS);

nexttile([1 1]);
spy(nfcount_strawman);
title('SAM method with \textbf{Lipschitz estimates } to determine probabilities', 'interpreter','latex','FontSize', FS)
yticks([1 2 3 4]);
xlabel('$k$','interpreter','latex', 'FontSize', FS);
ylabel('$I_k$', 'interpreter','latex', 'FontSize', FS); 

nexttile([1 1]);
spy(nfcount(:,1:100));
title('SAM method with \textbf{uniform sampling } to determine probabilities', 'interpreter','latex', 'FontSize', FS)
yticks([1 2 3 4]);
xlabel('$k$','interpreter','latex', 'FontSize', FS);
ylabel('$I_k$', 'interpreter','latex', 'FontSize', FS); 

t.TileSpacing = 'compact';
t.Padding = 'compact';

saveas(gcf,'failure.eps','epsc');





