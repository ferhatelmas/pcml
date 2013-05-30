function lr_plotter(v, tr_err_avg, val_err_avg, std_devs, v_opt, v_opt_i)
%LR_PLOTTER(v, tr_err_avg, val_err_avg, std_devs)
% plots linear regression validation errors, mean & standard deviation
% v: vector of Tikhonov regularizers
% val_err_avg: vector of cross validation error averages for parameters v
% tr_err_avg: vector of cross validation training error averages for parameters v
% std_devs: vector of standard deviations of M-folds for each v
% v_opt: optimum v selected by cross-validation

close all
figure(1);
semilogx(v,tr_err_avg,'-r');
hold on
semilogx(v,val_err_avg,'-g');
hold on
semilogx(v,val_err_avg + std_devs,'--g');
hold on
semilogx(v_opt,val_err_avg(v_opt_i),'ob'); % mark optimum v on plot
legend('Training','Validation','Standard Deviation','v*');
title('Training, Validation and Std for Cross Validation');
xlabel('v');
ylabel('Error');