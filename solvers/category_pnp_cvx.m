function [result] = category_pnp_cvx(input_problem)
%CATEGORY_PNP_CVX Use CVX to solve 2D-3D directly

N = input_problem.N;
K = input_problem.K;
K_intrinsic = input_problem.K_intrinsic;
kpts_I = input_problem.kpts_I;
bearing_vectors = input_problem.bearing_vectors;
cad_db_array = input_problem.cad_db_array;
R_guess = input_problem.R_guess;
t_guess = input_problem.t_guess;
c_guess = input_problem.c_guess;



end

