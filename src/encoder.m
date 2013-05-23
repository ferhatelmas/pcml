function T_T = encoder(t,k)
%ENCODER(t)
% t: target vector
% k: class size
% build 1-of-K encoding matrix of targets

n = length(t); % sample size

li = 0:k:(n-1)*k;
T_T = zeros(k,n);
T_T(li+t) = 1; %T_T(k,n) = 1 if t(n) = k, 0 o.w.