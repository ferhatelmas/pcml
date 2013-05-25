importfile('../data/norb_5class.mat');

n = size(train_left_s,2);

m_l = mean(train_left_s, 2);
m_rep = repmat(m_l, 1, n);
temp = (train_left_s - m_rep) .^ 2;
dev = sqrt(sum(temp, 2) ./ n);
istd_l = 1 ./ dev;

m_r = mean(train_right_s, 2);
m_rep = repmat(m_r, 1, n);
temp = (train_right_s - m_rep) .^ 2;
dev = sqrt(sum(temp, 2) ./ n);
istd_r = 1 ./ dev;

X_L = train_left_s;
X_R = train_right_s;
T = train_cat_s;

X_L_test = normalize(test_left_s, m_l, istd_l);
X_R_test = normalize(test_right_s, m_r, istd_r);
T_test = test_cat_s;

clear train_left_s train_right_s train_cat_s test_cat_s test_left_s test_right_s m_l m_r istd_l istd_r n dev m_rep temp
