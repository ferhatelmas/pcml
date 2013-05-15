function plotter(tr_err, val_err, tn)
%PLOTTER(TR_ERR, VAL_ERR)
% plots training and validation errors on the same figure
% tr_err: vector of training errors for an epoch
% val_err: vector of validation errors for an epoch
% tn: trial number
% n: number of instances used for training and validation

if(tn == 1) % initialize graph object in first trial
    close all
    figure(1);
    plot(tn,tr_err,'*b');
    hold on
    plot(tn,val_err,'*g');
    legend('Training','Validation');
    title('Training and Validation Errors for MLP Training');
    xlabel('Trial #');
    ylabel('Logistic Error');
else    
    figure(1);
    plot(tn,tr_err,'*b');
    hold on
    plot(tn,val_err,'*g');
end

xlim([0 tn+1]);
ylim([0 1.0]); % upper-limit is random and can be changed later
hold on