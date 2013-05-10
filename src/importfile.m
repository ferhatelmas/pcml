function importfile(fileToRead)
%IMPORTFILE(fileToRead)
%  Imports data from the specified mat file
%  fileToRead:  file to read

% Import the file
newData1 = load('-mat', fileToRead);

% Create new variables in the base workspace from those fields
vars = fieldnames(newData1);
for i = 1:length(vars)
    assignin('base', vars{i}, double(newData1.(vars{i})));
end

