function splitdata(dataset,filename)
%SPLITDATA(dataset)
%  Splits training dataset into two as training (2/3) and validation (1/3)
%  and saves them into a mat file
%  dataset:  matrix holding the training data

num_instances = size(dataset,2);
% Create random permutation with 1/3 of total number of instances
val_columns = randperm(num_instances, num_instances*1/3);
% Use these random numbers to pick instances for validation dataset
val_data = dataset(:,val_columns);

% use rest of instances for training
dataset(:,val_columns) = [];
train_data = dataset;

% save validation and training datasets to file for later use
save(strcat('val_', filename,'.mat'), 'val_data');
save(strcat('train_', filename, '.mat'), 'train_data');

