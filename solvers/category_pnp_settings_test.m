rng('default')

trials = 30;
N = 66; % num keypoints
K = 79; % num models
t_errs = zeros(trials, 1);

for trial_count=1:trials
    % generate random category problem data
    gt_weights = abs(randn(1, 1, K));
    gt_weights = gt_weights / sum(gt_weights);
    cad_db_array = randn(3, N, K);
    gt_cad = sum(multiprod(gt_weights, cad_db_array), 3);
    gt_R = randrot(3);
    t_scale = 50;
    gt_t = abs(randn(3, 1)) * t_scale;

    K_intrinsic = [2.3045479e+01, 0, 1.6862379e+02;
                   0, 2.3058757e+01, 1.3549849e+02;
                   0, 0, 1];


    % measurements
    cad_W = gt_R * gt_cad + gt_t;
    kpts_I = K_intrinsic * cad_W;
    kpts_I = kpts_I(1:2,:);
    bearing_vectors = cad_W ./ vecnorm(cad_W);

    % perturb guesses
    perturb_amp = 10;
    t_perturbed = gt_t + perturb_amp * randn(3,1);

    problem.bearing_vectors = bearing_vectors;
    problem.kpts_I = kpts_I;
    problem.K_intrinsic = K_intrinsic;
    problem.cad_db_array    = cad_db_array;
    problem.N               = N;
    problem.K               = K;
    problem.lambda          = 0;

    % solve
    problem.R_guess         = gt_R;
    problem.t_guess         = gt_t;
    problem.c_guess         = squeeze(gt_weights);
    [x_gt_init, xcost_gt_init] = category_pnp(problem);

    % perturbed solve
    problem.R_guess         = gt_R;
    problem.t_guess         = t_perturbed;
    problem.c_guess         = squeeze(gt_weights);
    [x_perturbed_init, xcost_perturbed_init] = category_pnp(problem);

    % c_normalized = x_perturbed_init.c ./ sum(x_perturbed_init.c);
    % R = x_perturbed_init.R;
    % problem.R = R;
    % problem.c = c_normalized;
    % problem.t_guess = x_perturbed_init.t;
    % [x_t_refined, xcost_t_refined] = category_pnp_t_only(problem);

    t_errs(trial_count) = norm(x_perturbed_init.t - gt_t);
end
    
fprintf('t gt init  : %.2f, %.2f, %.2f\n', x_gt_init.t)
fprintf('t pert init: %.2f, %.2f, %.2f\n', x_perturbed_init.t)
fprintf('t pert err : %.2f\n', t_perturbed_err)
fprintf('t gt       : %.2f, %.2f, %.2f\n', gt_t)
fprintf('------------------------------\n')
% fprintf('c gt init  : %.2f, %.2f, %.2f\n', x_gt_init.c)
% fprintf('c pert init: %.2f, %.2f, %.2f\n', x_perturbed_init.c)
% fprintf('c gt       : %.2f, %.2f, %.2f\n', gt_weights)
