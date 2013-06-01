function matrix = initialize_weights(m, n)
%INITIALIZE_WEIGHTS(m, n)
% initializes a matrix with given size
% to make gradient descent faster 
% m: output size
% n: input size

matrix = normrnd(0, sqrt(1/(100*n)), m, n);
% matrix = 0.1*ones(m,n);