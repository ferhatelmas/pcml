function splitT(T,filename,val_columns)
%SPLITT(T,filename)
%  Splits training dataset into two as training (2/3) and validation (1/3)
%  and saves them into a mat file
%  dataset:  matrix holding the training data

% Use these random numbers to pick instances for validation dataset
t_val = T(:,val_columns); 

% use rest of instances for training
T(:,val_columns) = [];
t_train = T;

% save validation and training datasets to file for later use
save(strcat('../data/val_', filename,'.mat'), 't_val');
save(strcat('../data/train_', filename, '.mat'), 't_train');