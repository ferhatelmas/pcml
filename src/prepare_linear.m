importfile('../data/norb_5class.mat');

X = [train_left_s; train_right_s];
[m, istd] = find_par(X);

X_test = normalize([test_left_s; test_right_s], m, istd);

T = train_cat_s;
T_test = test_cat_s;

clear train_left_s train_right_s train_cat_s test_cat_s test_left_s test_right_s m istd notes