importfile('../data/norb_5class.mat');

n = size(train_left_s,2);

[m_l, istd_l] = find_par(train_left_s);
[m_r, istd_r] = find_par(train_right_s);

X_L = train_left_s;
X_R = train_right_s;
T = train_cat_s;

X_L_test = normalize(test_left_s, m_l, istd_l);
X_R_test = normalize(test_right_s, m_r, istd_r);
T_test = test_cat_s;

clear train_left_s train_right_s train_cat_s test_cat_s test_left_s test_right_s m_l m_r istd_l istd_r n dev m_rep temp
