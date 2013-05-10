function e_log = logerr(t, a_3)
%ERROR: calculates logistic error for an input vector x
% t: target value vectors for input x
% a_3: MLP 3rd activation level output vector for inputs in x
% e_log: logistic error for whole batch

a = -t.*a_3
e_log = (a<0).*log1p(exp(a)) + (a>=0).*log(1+exp(a));

% number of instances used in an epoch for training and validation datasets
n = length(t);

% calculate total error by averaging over error for each instance in epoch 
e_log = sum(e_log)/n;
