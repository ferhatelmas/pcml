function multi_plotter(tr_err_1, tr_err_2, val_err_1, val_err_2, zero_one_1, zero_one_2)
%MULTI_PLOTTER(tr_err_1, tr_err_2, val_err_1, val_err_2, zero_one_1, zero_one_2)
% plots two trials on the same figure
% tr_err_i: vector of training errrors for each epoch for trial i
% val_err_i: vector of validation errors for each epoch for trial i
% zero_one_i: vector of zero-one errors for each epoch for trial i

close all
figure(1);
plot(tr_err_1,'-r');
hold on
plot(val_err_1,'-g');
hold on
plot(zero_one_1,'-k');
hold on
plot(tr_err_2,'--r');
hold on
plot(val_err_2,'--g');
hold on
plot(zero_one_2,'--k');

legend('Training 1','Validation 1','Zero-One 1','Training 2','Validation 2','Zero-One 2');
title('Training, Validation and Zero-One Errors for MLP Training');
xlabel('Epoch #');
ylabel('Error Value');
