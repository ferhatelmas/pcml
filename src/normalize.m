function normalized_data = normalize(dataset, m, istd)
%NORMALIZE(dataset, m, istd)
% Normalizes given data and saves them into mat files
% dataset: matrix holding the data
% m: mean
% istd: inverse standard deviation

n = size(dataset, 2);

% calculate mean and std_deviation of dataset
m = repmat(m, 1, n);
istd = repmat(istd, 1, n);

% standardize by subtracting mean and dividing by standard deviation
normalized_data = istd .* (dataset - m);