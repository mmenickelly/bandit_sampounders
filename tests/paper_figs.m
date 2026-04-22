% performance profiles 
FS = 18; 

load lipschitz_test_1.mat
Hplot = H; effortplot = effort; 
num_seeds = 3;
np = 53;
for j = 2:num_seeds
    filestr = strcat('lipschitz_test_', num2str(j),'.mat');
    load(filestr);
    Hplot = cat(2, Hplot, H);
    effortplot = cat(2, effortplot, effort);
    Hplot(:,((j-1)*np+1):(j*np),1) = Hplot(:,1:np,1); % duplicate pounders entries
    effortplot(:,((j-1)*np+1):(j*np),1) = effortplot(:,1:np,1); % duplicate pounders entries
end
H = Hplot;
effort = effortplot; 

logplot = 0;
figure;
perf_profile_budget(H, effort, 1e-3, logplot, num_seeds);
legend('POUNDERS', 'SAM-POUNDERS, Uniform sampling only', 'SAM-POUNDERS, Lipschitz estimation only', 'SAM-POUNDERS, Uniform + Lipschitz', 'FontSize', FS, 'Location', 'SouthEast');
title('$\tau=10^{-3}$','interpreter','latex','FontSize',FS);
saveas(gcf, 'lipschitz_3.eps', 'epsc');

figure;
perf_profile_budget(H, effort, 1e-7, logplot, num_seeds);
legend('POUNDERS', 'SAM-POUNDERS, Uniform sampling only', 'SAM-POUNDERS, Lipschitz estimation only', 'SAM-POUNDERS, Uniform + Lipschitz', 'FontSize', FS, 'Location', 'SouthEast');
title('$\tau=10^{-7}$','interpreter','latex','FontSize',FS);
saveas(gcf, 'lipschitz_7.eps', 'epsc');

load surrogate_test.mat
Hplot_surr = H; effortplot_surr = effort; 
num_seeds = 3;
np = 53;
for j = 2:num_seeds
    filestr = strcat('surrogate_test_', num2str(j),'.mat');
    load(filestr);
    Hplot_surr = cat(2, Hplot_surr, H);
    effortplot_surr = cat(2, effortplot_surr, effort);
    Hplot_surr(:,((j-1)*np+1):(j*np),1) = Hplot_surr(:,1:np,1); % duplicate pounders entries
    effortplot_surr(:,((j-1)*np+1):(j*np),1) = effortplot_surr(:,1:np,1); % duplicate pounders entries
    Hplot_surr(:,((j-1)*np+1):(j*np),2) = Hplot(:,(j-1)*np+1:(j*np),2); % duplicate uniform entries
    effortplot_surr(:,((j-1)*np+1):(j*np),2) = effortplot(:,(j-1)*np+1:(j*np),2); % duplicate uniform entries
end
H = Hplot_surr;
effort = effortplot_surr; 
figure;
perf_profile_budget(H, effort, 1e-3, logplot, num_seeds);
legend('POUNDERS', 'SAM-POUNDERS, Uniform sampling only', 'SAM-POUNDERS, Surrogate estimation only', 'SAM-POUNDERS, Uniform + Surrogate', 'FontSize', FS, 'Location', 'SouthEast');
title('$\tau=10^{-3}$','interpreter','latex','FontSize',FS);
saveas(gcf, 'surrogate_3.eps', 'epsc');

figure;
perf_profile_budget(H, effort, 1e-7, logplot, num_seeds);
legend('POUNDERS', 'SAM-POUNDERS, Uniform sampling only', 'SAM-POUNDERS, Surrogate estimation only', 'SAM-POUNDERS, Uniform + Surrogate', 'FontSize', FS, 'Location', 'SouthEast');
title('$\tau=10^{-7}$','interpreter','latex','FontSize',FS);
saveas(gcf, 'surrogate_7.eps', 'epsc');

