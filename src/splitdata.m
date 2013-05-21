function [train_data, val_data, m, inverse_dev] = splitdata(dataset, val_columns)
%SPLITDATA(dataset)
%  Splits training dataset into two as training (2/3) and validation (1/3)
%  and saves them into a mat file
%  dataset:  matrix holding the training data

% Use these random numbers to pick instances for validation dataset
val_data = dataset(:, val_columns);

% use rest of instances for training
dataset(:, val_columns) = [];
train_data = dataset;

n = size(train_data, 2);

% calculate mean and std_deviation of dataset
m = mean(train_data, 2);
m_rep = repmat(m, 1, n);
temp = (train_data - m_rep) .^ 2;
dev = sqrt(sum(temp, 2) ./ n);
inverse_dev = 1 ./ dev;