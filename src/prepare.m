importfile('../data/norb_binary.mat');

train_cat_s = train_cat_s - 2;
test_cat_s = test_cat_s - 2;
num_instances = size(train_left_s,2);

% Create random permutation with 1/3 of total number of instances
val_columns = randperm(num_instances, num_instances*1/3);
[X_L, X_L_val, m_l, istd_l] = splitdata(train_left_s, val_columns);
[X_R, X_R_val, m_r, istd_r] = splitdata(train_right_s, val_columns);
[T, T_val] = splitT(train_cat_s, val_columns);

clear train_left_s train_right_s train_data

X_L = normalize(X_L, m_l, istd_l);
X_R = normalize(X_R, m_r, istd_r);
X_L_val = normalize(X_L_val, m_l, istd_l);
X_R_val = normalize(X_R_val, m_r, istd_r);

X_L_test = normalize(test_left_s, m_l, istd_l);
X_R_test = normalize(test_right_s, m_r, istd_r);
T_test = test_cat_s;

clear t_train t_val test_cat_s test_left_s test_right_s train_cat_s val_data m_l m_r istd_l istd_r
