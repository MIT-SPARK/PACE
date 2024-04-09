function [x, xcost] = category_pnp_t_only(input_problem)
%CATEGORY_PNP Use manopt to solve category level PnP

N = input_problem.N;
K = input_problem.K;
K_intrinsic = input_problem.K_intrinsic;
kpts_I = input_problem.kpts_I;
bearing_vectors = input_problem.bearing_vectors;
cad_db_array = input_problem.cad_db_array;
t_guess = input_problem.t_guess;
R = input_problem.R;
c = reshape(input_problem.c, 1, 1, K);

% create the problem structure
M = euclideanfactory(3);

% cost 2: dist in calibrated image plane
function [f, store] = cost_image_dist(X, store)
    t = X;
    weighted_cad = sum(multiprod(c, cad_db_array), 3);
    weighted_cad_cam = R * weighted_cad + reshape(t, [3,1]);
    weighted_cad_image = weighted_cad_cam ./ weighted_cad_cam(3, :);
    measured_pts_image = bearing_vectors ./ bearing_vectors(3,:);
    %reproj_errs = vecnorm(weighted_cad_image - measured_pts_image).^2;
    diffs = weighted_cad_image - measured_pts_image;
    reproj_errs = diffs .* diffs;
    reproj_errs = sum(reproj_errs, 1);
    f = sum(reproj_errs);
end

problem.M = M;
problem.cost = @cost_image_dist;
%problem.egrad = @e_grad_image_dist;
problem = manoptAD(problem);

% Numerically check gradient consistency (optional).
%checkgradient(problem);

x0 = t_guess;

% Solve.
options.useRand = false;
[x, xcost, info, options] = trustregions(problem, x0, options);
 
end

