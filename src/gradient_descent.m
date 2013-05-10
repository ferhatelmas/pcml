function next = gradient_descent(previous, current, gradient, nu, mu)
%GRADIENT_DESCENT
% updates old to new using gradient descent with momentum
% old: old values of bias or weights to be updated (vector or matrix)
% nu: learning rate
% mu: momentum term

% calculate update in current iteration
delta_previous = current - previous;
delta_current = -nu*(1-mu)*gradient + mu*delta_previous;

% update current value with this difference
next = current + delta_current;