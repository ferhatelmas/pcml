function e_log = logerr(t, a_3)
%ERROR: calculates logistic error for an input vector x
% t_i: target value vectors for input x
% a_3_i: MLP 3rd activation level output vector for inputs in x
% e_log: logistic error vector

e_log = -log(sigmoid(-t.*a_3));
