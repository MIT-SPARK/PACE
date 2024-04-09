function [x, xcost] = category_pnp(input_problem, cost_only)
%CATEGORY_PNP Use manopt to solve category level PnP

N = input_problem.N;
K = input_problem.K;
K_intrinsic = input_problem.K_intrinsic;
kpts_I = input_problem.kpts_I;
bearing_vectors = input_problem.bearing_vectors;
cad_db_array = input_problem.cad_db_array;
R_guess = input_problem.R_guess;
t_guess = input_problem.t_guess;
c_guess = input_problem.c_guess;
lambda = input_problem.lambda;

% normalization constant
NC = 1e6;
NC = 1;

% create the problem structure
manifolds = struct();
manifolds.R = rotationsfactory(3);
manifolds.t = euclideanfactory(3);
manifolds.c = positivefactory(K);
M = productmanifold(manifolds);

% cost 0: point to line distance
function [f] = cost_p2l_dist(X)
    R = X.R;
    t = X.t;
    c = reshape(X.c, 1, 1, K);
    weighted_cad = sum(multiprod(c, cad_db_array), 3);
    weighted_cad_cam = R * weighted_cad + t;
    
    f       = 0;
    for i = 1:N
        bi  = bearing_vectors(:,i);
        ai  = weighted_cad_cam(:,i);
        
        f   = f + ai' * ( (eye(3) - bi*bi') * ai);
    end
end

% cost 1: angle
function [f] = cost_angular(X)
    R = X.R;
    t = X.t;
    c = reshape(X.c, 1, 1, K);
    weighted_cad = sum(multiprod(c, cad_db_array), 3);
    weighted_cad_cam = R * weighted_cad + t;
    weighted_cad_cam = weighted_cad_cam ./ vecnorm(weighted_cad_cam);
    reproj_errs = ones(1, N) - sum(weighted_cad_cam .* bearing_vectors, 1);
    f = sum(reproj_errs);
end

% cost 2: dist in calibrated image plane
function [f] = cost_image_dist(X)
    R = X.R;
    t = X.t;
    c = reshape(X.c, 1, 1, K);
    weighted_cad = sum(multiprod(c, cad_db_array), 3);
    weighted_cad_cam = R * weighted_cad + t;
    weighted_cad_image = weighted_cad_cam ./ weighted_cad_cam(3, :) * NC;
    measured_pts_image = bearing_vectors ./ bearing_vectors(3,:) * NC;
    %reproj_errs = vecnorm(weighted_cad_image - measured_pts_image).^2;
    diffs = weighted_cad_image - measured_pts_image;
    reproj_errs = diffs .* diffs;
    reproj_errs = sum(reproj_errs, 1);
    f = sum(reproj_errs) + lambda * sum(c.*c);
end

% cost 3: in pixels
function [f] = cost_image_pixel(X)
    R = X.R;
    t = X.t;
    c = reshape(X.c, 1, 1, K);
    weighted_cad = sum(multiprod(c, cad_db_array), 3);
    weighted_cad_cam = R * weighted_cad + t;
    weighted_cad_I = K_intrinsic * weighted_cad_cam;
    weighted_cad_I = weighted_cad_I ./ weighted_cad_I(3, :);
    
    % debug messages
%     fprintf('\nR: \n%d \n', R)
%     fprintf('\nt: \n%d \n', t)
%     fprintf('\nc: \n%d \n', c)
%     fprintf('\nK_intrinsic: \n%d \n', K_intrinsic)
%     fprintf('\nkpts_I: \n%d \n', kpts_I)
%     fprintf('\nweighted_cad: \n%d \n', weighted_cad_I)

%     dlmwrite('/home/jnshi/Desktop/R.csv', R, ',') 
%     dlmwrite('/home/jnshi/Desktop/t.csv', t, ',')
%     dlmwrite('/home/jnshi/Desktop/c.csv', c, ',')    
%     dlmwrite('/home/jnshi/Desktop/K_intrinsic.csv', K_intrinsic, ',')
%     dlmwrite('/home/jnshi/Desktop/kpts_I.csv',kpts_I,',')
%     dlmwrite('/home/jnshi/Desktop/weighted_cad_I.csv',weighted_cad_I,',')

%     fprintf('\nWeighted_cad_I first: \n%f,\n %f,\n %f\n', weighted_cad_I(1,1), ...
%         weighted_cad_I(2,1),weighted_cad_I(3,1))
%     fprintf('\nWeighted_cad_I last: \n%f,\n %f,\n %f\n', weighted_cad_I(1,end), ...
%     weighted_cad_I(2,end),weighted_cad_I(3,end))

    diffs = kpts_I - weighted_cad_I(1:2, :);
%     dlmwrite('/home/jnshi/Desktop/diffs.csv',diffs,',')

    reproj_errs = diffs .* diffs;
    reproj_errs = sum(reproj_errs, 1);
    f = sum(reproj_errs);
end

% gradient 1: for cost 1
function [egrad] = e_grad_angular(X)
    R = X.R;
    t = X.t;
    c = reshape(X.c, 1, 1, K);
    
    weighted_cad = sum(multiprod(c, cad_db_array), 3);
    weighted_cad_cam = R * weighted_cad + t;
    weighted_cad_cam_norms = vecnorm(weighted_cad_cam);
    
    R_grad = zeros(3,3);
    t_grad = zeros(3,1);
    c_grad = zeros(K,1);
    for i=1:N
        B = squeeze(cad_db_array(:, i, :));
        
        % R grad
        part_A = (bearing_vectors(:, i) * weighted_cad(:,i)') ...
            / weighted_cad_cam_norms(i);
        part_B = (bearing_vectors(:, i)' * weighted_cad_cam(:,i) ...
            * weighted_cad_cam(:,i) * weighted_cad(:,i)') ...
            / weighted_cad_cam_norms(i)^3;
        R_grad = R_grad - (part_A - part_B);
        
        % t grad
        part_A = (bearing_vectors(:, i)) ...
            / weighted_cad_cam_norms(i);
        part_B = (bearing_vectors(:, i)' * weighted_cad_cam(:,i) ...
            * weighted_cad_cam(:,i)) ...
            / weighted_cad_cam_norms(i)^3;
        t_grad = t_grad - (part_A - part_B);
        
        % c grad
        part_A = (B' * R' * bearing_vectors(:, i)) ...
            / weighted_cad_cam_norms(i);
        part_B = (bearing_vectors(:, i)' * weighted_cad_cam(:,i) ...
            * B' * R' * weighted_cad_cam(:,i)) ...
            / weighted_cad_cam_norms(i)^3;
        c_grad = c_grad - (part_A - part_B);
    end
    egrad.R = R_grad;
    egrad.t = t_grad;
    egrad.c = c_grad;
end


% gradient 2: for cost 2 (dist on image plane)
function [egrad] = e_grad_image_dist(X)
    R = X.R;
    t = X.t;
    c = X.c;
    
    R_grad = zeros(3,3);
    t_grad = zeros(3,1);
    c_grad = zeros(K,1);
    for i=1:N
        % B: 3-by-K matrix describing the ith keypoint across all models
        B = squeeze(cad_db_array(:, i, :));
        % p: ith measured point (in homogenous coordinate)
        p = bearing_vectors(:, i);
        p = p ./ p(3);
        % convenience variables
        p1 = p(1);
        p2 = p(2);
        R1 = R(1, :)';
        R2 = R(2, :)';
        R3 = R(3, :)';
        t1 = t(1);
        t2 = t(2);
        t3 = t(3);

        % R grad
        R1_grad = -2 * (p1 - (R1' * B * c + t1)/(R3' * B * c + t3)) ...
            * (B * c) / (R3' * B * c + t3);
        R2_grad = -2 * (p2 - (R2' * B * c + t2)/(R3' * B * c + t3)) ...
            * (B * c) / (R3' * B * c + t3);
        R3_grad = 2 * (R1' * B * c + t1) ...
            * (p1 - (R1' * B * c + t1)/(R3' * B * c + t3)) ...
            * (B * c) / (R3' * B * c + t3)^2 ...
            + 2 * (R2' * B * c + t2) ...
            * (p2 - (R2' * B * c + t2)/(R3' * B * c + t3)) ...
            * (B * c) / (R3' * B * c + t3)^2;
        R_grad = [R1_grad'; R2_grad'; R3_grad'];
        
        % t grad
        t_grad(1) = -(2 * ( p1 - (t1 + R1' * B * c) / (t3 + R3' * B * c)))...
            / (t3 + R3' * B * c);
        t_grad(2) = -(2 * ( p2 - (t2 + R2' * B * c) / (t3 + R3' * B * c)))...
            / (t3 + R3' * B * c);
        t_grad(3) = (2 * (t1 + R1' * B * c) ...
            * (p1 - (t1 + R1' * B * c) / (t3 + R3' * B * c))) ...
            / (t3 + R3' * B * c)^2 ...
            + (2 * (t2 + R2' * B * c) ...
            * (p2 - (t2 + R2' * B * c) / (t3 + R3' * B * c))) ...
            / (t3 + R3' * B * c)^2;
        
        % c grad
        c_grad = -((2*(p1-(t1+c'*B'*R1)/(t3+c'*B'*R3)))/(t3+c'*B'*R3)*B'*R1...
            -(2*(t1+c'*B'*R1)*(p1-(t1+c'*B'*R1)/(t3+c'*B'*R3)))/(t3+c'*B'*R3).^2*B'*R3...
            +(2*(p2-(t2+c'*B'*R2)/(t3+c'*B'*R3)))/(t3+c'*B'*R3)*B'*R2...
            -(2*(t2+c'*B'*R2)*(p2-(t2+c'*B'*R2)/(t3+c'*B'*R3)))/(t3+c'*B'*R3).^2*B'*R3);

    end
    egrad.R = R_grad;
    egrad.t = t_grad;
    egrad.c = c_grad;
end

problem.M = M;
problem.cost = @cost_image_dist;
%problem.cost = @cost_image_pixel;
% problem.egrad = @e_grad_angular;

if ~cost_only
    % Numerically check gradient consistency (optional).
    %checkgradient(problem);
    problem = manoptAD(problem);

    x0.R = R_guess;
    x0.t = t_guess;
    x0.c = c_guess;

    % Solve.
    options.useRand = false;
    [x, xcost, info, options] = trustregions(problem, x0, options);
else
    x.R = R_guess;
    x.t = t_guess;
    x.c = c_guess;
    xcost = cost_image_pixel(x);
    fprintf('\n x cost: %e\n', xcost);
end
 
end

