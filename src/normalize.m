function normalize(dataset, filename)
%NORMALIZE(data)
% Normalizes given data and saves them into mat files
% dataset: matrix holding the data
% filename: filename used to save dataset to file

n = size(dataset, 1);

% calculate mean and std_deviation of dataset
mean_of_dataset = repmat(mean(dataset), n, 1);
inverse_deviation_of_dataset = repmat(1 ./ std(dataset), n, 1);

% standardize by substracting mean and dividing by standard deviation
normalized_data = inverse_deviation_of_dataset .* (dataset - mean_of_dataset);

save(filename, 'normalized_data'); 