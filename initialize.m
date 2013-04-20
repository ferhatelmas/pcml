function matrix = initialize(m, n)
%INITIALIZE(m, n)
% initializes a matrix with given size
% to make gradient descent faster 
% m: output size
% n: input size
    
matrix = normrnd(0, sqrt(1/(1+n)), m, n);
