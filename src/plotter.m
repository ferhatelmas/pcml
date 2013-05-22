function plotter(tr_err, val_err, zerone_err, ec)
%PLOTTER(TR_ERR, VAL_ERR)
% plots training and validation errors on the same figure
% tr_err: vector of training errors for an epoch
% val_err: vector of validation errors for an epoch
% tn: epoch number
% n: number of instances used for training and validation

if(ec == 1) % initialize graph object in first trial
    close all
    figure(1);
    plot(ec,tr_err(ec),'*r');
    hold on
    plot(ec,val_err(ec),'*g');
    hold on
    plot(ec,zerone_err(ec),'*k');
    legend('Training','Validation','0-1');
    title('Training, Validation and Zero-One Errors for MLP Training');
    xlabel('Epoch #');
    ylabel('Error Value');
else    
    figure(1);
    plot(ec,tr_err(ec),'*r');
    hold on
    plot(ec,val_err(ec),'*g');
    hold on
    plot(ec,zerone_err(ec),'*k');
end

xlim([0 ec+1]);
%ylim([0 1.1]); % upper-limit is random and can be changed later
hold on