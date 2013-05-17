importfile('../data/norb_binary.mat');
train_cat_s = train_cat_s - 2;
num_instances = size(train_left_s,2);
% Create random permutation with 1/3 of total number of instances
val_columns = randperm(num_instances, num_instances*1/3);
splitdata(train_left_s, 'train_left_s',val_columns);
splitdata(train_right_s, 'train_right_s',val_columns);
splitT(train_cat_s, 't',val_columns);
importfile('../data/train_train_left_s.mat');
X_L = train_data;
importfile('../data/train_train_right_s.mat');
X_R = train_data;
importfile('../data/val_train_left_s.mat');
X_L_val = val_data;
importfile('../data/val_train_right_s.mat');
X_R_val = val_data;
importfile('../data/train_t.mat');
T = t_train;
importfile('../data/val_t.mat');
T_val = t_val;

clear t_train t_val test_cat_s test_left_s test_right_s train_cat_s train_data train_left_s train_right_s val_data