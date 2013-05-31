function multi_plotter(tr_err1, tr_err2, tr_err3, val_err1, val_err2, val_err3, zero_one_err1, zero_one_err2, zero_one_err3)
%MULTI_semilogyTER(tr_err_1, tr_err_2, val_err_1, val_err_2, zero_one_err_1, zero_one_err_2)
% semilogys two trials on the same figure
% tr_err_i: vector of training errrors for each epoch for trial i
% val_err_i: vector of validation errors for each epoch for trial i
% zero_one_err_i: vector of zero-one errors for each epoch for trial i

close all
figure(1);
semilogy(tr_err1,'-r');
hold on
semilogy(val_err1,'--r');
hold on
semilogy(zero_one_err1,'or');
hold on
semilogy(tr_err2,'-b');
hold on
semilogy(val_err2,'--b');
hold on
semilogy(zero_one_err2,'ob');
hold on
semilogy(tr_err3,'-k');
hold on
semilogy(val_err3,'--k');
hold on
semilogy(zero_one_err3,'ok');

legend('Training 1','Validation 1','Zero-One 1','Training 2','Validation 2','Zero-One 2','Training 3','Validation 3','Zero-One 3');
title('Training, Validation and Zero-One Errors for MLP Training');
xlabel('Epoch #');
ylabel('Error Value');
