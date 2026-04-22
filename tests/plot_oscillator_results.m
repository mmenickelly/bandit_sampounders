% make plots
load('oscillator_results.mat')
num_seeds = 30;


figure;
colors = ['b', 'r', 'g', 'm'];
markers = ['s' 'o' '^' 'v' 'p' '<' 'x' 'h' '+' 'd' '*' '<'];
LW = 2; MW = 5; FS = 18;
lower_bound = min(min(min(H)));

handle_array = [];
for solver = 1:4
    for seed = 1:num_seeds
        array = effort(:,solver,seed);
        array = array(~isnan(array));
        X{seed} = array;
        array = H(:, solver, seed); 
        array = log10(array(~isnan(array)) - lower_bound + eps);
        for t = 2:length(array)
            array(t) = min(array(t-1), array(t));
        end
        Y{seed} = array;
    end

    nCurves = num_seeds;

    xmin = max(cellfun(@(x) min(x), X));   % overlap start
    xmax = min(cellfun(@(x) max(x), X));   % overlap end

    xgrid = linspace(xmin, xmax, 200);

    Ygrid = nan(nCurves, length(xgrid));

    for i = 1:nCurves
        Ygrid(i,:) = interp1(X{i}, Y{i}, xgrid, 'linear', NaN);
    end

    q50 = median(Ygrid, 1, 'omitnan');
    q25 = quantile(Ygrid, 0.25, 1);
    q75 = quantile(Ygrid, 0.75, 1);
    q10 = quantile(Ygrid, 0.10, 1);
    q90 = quantile(Ygrid, 0.90, 1);

    hold on;

    % 25-75 band
    % fill([xgrid fliplr(xgrid)], ...
    %      [q25 fliplr(q75)], ...
    %      colors(solver), ...
    %      'EdgeColor','none', ...
    %      'FaceAlpha',0.5);
    
    % 10-90 band
    fill([xgrid fliplr(xgrid)], ...
         [q10 fliplr(q90)], ...
         colors(solver), ...
         'EdgeColor','none', ...
         'FaceAlpha',0.3);
    
    % Median
    handle_array(solver) = plot(xgrid, q50, 'LineWidth', LW, 'Color', colors(solver), 'Marker', markers(solver), 'MarkerSize', MW);

    plot(xgrid,q10,'Color','k','LineStyle','--','LineWidth',0.75)
    plot(xgrid,q90,'Color','k','LineStyle','--','LineWidth',0.75)
    grid on;
end
xlabel('Component function evaluations','FontSize',FS)
ylabel('$\log_{10}(h(F(x)) - h(F(x^{true})))$','interpreter','latex','FontSize',FS)

solver_names = {'POUNDERS','SAM-POUNDERS with Gemini Pro 3.1 sampling', 'SAM-POUNDERS with uniform random sampling', 'SAM-POUNDERS with Lipschitz estimation sampling'};
legend(handle_array, solver_names,'FontSize',FS); 

xlim([200 768.492])
saveas(gcf,'gemini.eps','epsc')