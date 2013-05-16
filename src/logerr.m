function e_log = logerr(t, a_3)
%ERROR: calculates logistic error for an input vector x
% t: target value vectors for input x
% a_3: MLP 3rd activation level output vector for inputs in x
% e_log: logistic error for whole batch

a = -t.*a_3;
e_log = mean((a<0).*log1p(exp(a)) + (a>=0).*log(1+exp(a)));
