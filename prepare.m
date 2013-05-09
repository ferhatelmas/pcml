importfile('norb_binary.mat');
train_cat_s = train_cat_s - 2;
splitdata(train_left_s, 'train_left_s');
splitdata(train_right_s, 'train_right_s');