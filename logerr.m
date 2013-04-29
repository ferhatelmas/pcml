function e_log = logerr(t_i, a_3_i)
%ERROR: calculates logistic error for a single input
% t_i: class value for input x_i
% a_3_i: MLP 3rd activation level output for input x_i
% e_log: logistic error value

e_log = -log(sigmoid(-t_i*a_3_i));
