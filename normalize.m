function normalize(dataset, filename)
%NORMALIZE(data)
% Normalizes given data and saves them into mat files
% dataset: matrix holding the data

n = size(dataset, 1);

mean_of_dataset = repmat(mean(dataset), n, 1);
inverse_deviation_of_dataset = repmat(1 ./ std(dataset), n, 1);

normalized_data = inverse_deviation_of_dataset .* (dataset - mean_of_dataset);

save(filename, 'normalized_data'); 