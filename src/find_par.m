function [m istd] = find_par(data)
%FIND_VAR(data)
% Find mean and deviation
% calculate mean and std_deviation of dataset
% m: Mean throuhgout the rows
% istd: Inverse standard deviation

m = mean(data, 2);
avg_rep = repmat(m, 1, n);
temp = (train_data - avg_rep) .^ 2;
dev = sqrt(sum(temp, 2) ./ n);
istd = 1 ./ dev;
end