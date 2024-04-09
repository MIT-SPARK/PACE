cvx_solver mosek
m = 16; n = 8;
A = randn(m,n);
b = randn(m,1);

K = 3;
N = 30;
bearing_vectors = randn(3, N);

cvx_begin
    % d: distance along the rays
    variable d(N)
    % R_k = c_k * R
    variable R(3, 3, K)
    % t: translation
    variable t(3)
    % Sd = [d1 * bv1, ...]
    Sd = bearing_vectors .* d;
cvx_end