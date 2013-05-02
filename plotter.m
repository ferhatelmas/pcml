function plotter(tr_err, val_err, tn)
%PLOTTER(TR_ERR, VAL_ERR)
% plots training and validation errors on the same figure
% tr_err: vector of training errors for an epoch
% val_err: vector of validation errors for an epoch
% tn: trial number
% n: number of instances used for training and validation

% number of instances used in an epoch for training and validation datasets
n_tr = length(tr_err);
n_val = length(val_err);

% calculate total error by averaging over error for each instance in epoch 
tr_err = sum(tr_err)/n_tr;
val_err = sum(val_err)/n_val;

if(tn == 1) % initialize graph object in first trial
    figure(1);
    plot(tn,tr_err,'ob');
    hold on
    plot(tn,val_err,'og');
    legend('Training','Validation');
    title('Training and Validation Errors for MLP Training');
    xlabel('Trial #');
    ylabel('Logistic Error');
else    
    figure(1);
    plot(tn,tr_err,'ob');
    hold on
    plot(tn,val_err,'og');
end

xlim([0 tn+1]);
ylim([0 100]); % upper-limit is random and can be changed later
hold on


    
    