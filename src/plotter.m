function plotter(tr_err, val_err, zerone_err, ec)
%PLOTTER(tr_err, val_err, zerone_err, ec)
% plots training, validation and zero_one errors on the same figure
% tr_err: vector of training errors
% val_err: vector of validation errors
% zerone_err: vector of zero-one errors
% ec: epoch number

if(ec == 1) % initialize graph object in first trial
    close all
    figure(1)
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
    figure(1)
    plot(ec,tr_err(ec),'*r');
    hold on
    plot(ec,val_err(ec),'*g');
    hold on
    plot(ec,zerone_err(ec),'*k');
end

xlim([0 ec+1]);
%ylim([0 1.1]); % upper-limit is random and can be changed later
hold on