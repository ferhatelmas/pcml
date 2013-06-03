function [train_data, val_data, m, istd] = splitX(dataset, val_columns)
%SPLITX(dataset)
%  Splits training dataset into two as training (2/3) and validation (1/3)
%  and saves them into a mat file
%  dataset:  matrix holding the training data

% Use these random numbers to pick instances for validation dataset
val_data = dataset(:, val_columns);

% use rest of instances for training
dataset(:, val_columns) = [];
train_data = dataset;

% calculate mean and std_deviation of dataset
[m, istd] = find_parameters(train_data);