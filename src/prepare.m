importfile('../data/norb_binary.mat');
train_cat_s = train_cat_s - 2;
num_instances = size(train_left_s,2);
% Create random permutation with 1/3 of total number of instances
val_columns = randperm(num_instances, num_instances*1/3);
[m_l, istd_l] = splitdata(train_left_s, 'train_left_s',val_columns);
[m_r, istd_r] = splitdata(train_right_s, 'train_right_s',val_columns);
splitT(train_cat_s, 't',val_columns);
importfile('../data/train_train_left_s.mat');
X_L = train_data;
importfile('../data/train_train_right_s.mat');
X_R = train_data;

clear train_left_s train_right_s train_data

importfile('../data/val_train_left_s.mat');
X_L_val = normalize(val_data, m_l, istd_l);
importfile('../data/val_train_right_s.mat');
X_R_val = normalize(val_data, m_r, istd_r);
importfile('../data/train_t.mat');
T = t_train;
importfile('../data/val_t.mat');
T_val = t_val;

X_L_test = normalize(test_left_s, m_l, istd_l);
X_R_test = normalize(test_right_s, m_r, istd_r);
T_test = test_cat_s - 2;

clear t_train t_val test_cat_s test_left_s test_right_s train_cat_s val_data