function [m, istd] = find_parameters(data)
%FIND_PARAMETERS(data)
% Find mean and deviation
% calculate mean and std_deviation of dataset
% m: Mean throuhgout the rows
% istd: Inverse standard deviation


n = size(data,2);

m = mean(data, 2);
avg_rep = repmat(m, 1, n);
temp = (data - avg_rep) .^ 2;
dev = sqrt(sum(temp, 2) ./ n);
istd = 1 ./ dev;

end