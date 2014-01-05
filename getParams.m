function params = getParams(Z, numOfOmigaY, numOfOmigaX)

[n, m] = size(Z);

[U, S, V] = svd(Z);
% the exact rank of MC-1 matrix that has the best performance
params.rank_1 = 16;

% the exact rank of MC-b matrix that has the best performance
params.rank_b = 85;

% relative coverge tolerance
params.tol = 1e-4;

% decay parameter for mu
params.eta = 0.01;

% max outer iteration
params.maxOuterItr = 500;

% max inner iteration
params.maxInnerItr = 100;

params.mus = S(1) * params.eta;

% final mu 
params.muf = 0.01;

% tradeoff parameter
params.lamda = 1;

% step size of matrix Z
params.tauz = 0.5;

% step size of bias vector b;
params.taub = 0.5;


end