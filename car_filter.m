function [ cnt_f ] = car_filter( cnt )
mu = mean(cnt,2);        
    for i=1:size(cnt,2)      % 2 indicates the columns of matrix
        xi = cnt(:,i);       % EEG of a single channel
        cnt_f(:,i) = xi-mu;  
    end
end

 