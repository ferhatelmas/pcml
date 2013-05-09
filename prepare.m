importfile('norb_binary.mat');
train_cat_s = train_cat_s - 2;
splitdata(train_left_s, 'train_left_s');
splitdata(train_right_s, 'train_right_s');
splitT(train_cat_s, 't');
importfile('train_train_left_s.mat');
X_L = train_data;
importfile('train_train_right_s.mat');
X_R = train_data;
importfile('val_train_left_s.mat');
X_L_val = val_data;
importfile('val_train_right_s.mat');
X_R_val = val_data;
importfile('train_t.mat');
T = t_train;
importfile('val_t.mat');
T_val = t_val;

clear t_train t_val test_cat_s test_left_s test_right_s train_cat_s train_data train_left_s train_right_s val_data