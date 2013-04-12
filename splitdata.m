function splitdata(dataset)
%SPLITDATA(train_data)
%  Splits training dataset into two as training (2/3) and validation (1/3)
%  and saves them into a mat file
%  dataset:  matrix holding the training data

[num_pixels num_instances] = size(dataset);
% Create random permutation with 1/3 of total number of instances
val_columns = randperm(num_instances, num_instances*1/3);
% Use these random numbers to pick instances for validation dataset
val_data = dataset(:,val_columns);

% use rest of instances for training
dataset(:,val_columns) = [];
train_data = dataset;

% save validation and training datasets to file for later use
save('val_dataset.m', val_data);
save('train_dataset.m', train_data);

