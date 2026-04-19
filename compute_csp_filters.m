function [w] = compute_csp_filters(data1,data2,mm)
%% Class 1
Rh=0;
for i=1:size(data1,3)
    x1=data1(:,:,i)';
    % step1: Normalize Data
    for n=1:size(x1,1)
        x1(n,:)=x1(n,:)-mean(x1(n,:));
    end
    % step2: Calculate Covariance of each single trial
    rh=(x1*x1')/trace((x1*x1')); 
    %rh=(x1*x1')/sum(diag((x1*x1')));
    Rh=Rh+rh;    % mean of covariance of all the trials (1/2)
end

    % step3: Calculate mean of Rh 
Rh=Rh/size(data1,3);  % mean of covariance of all the trials (2/2)

%% Class 2
Rf=0;
for i=1:size(data2,3)
    x2=data2(:,:,i)';
    % step1: Normalize Data
    for n=1:size(x2,1)
        x2(n,:)=x2(n,:)-mean(x2(n,:));
    end
    % step2: Calculate Covariance of each single trial
    rf=(x2*x2')/trace((x2*x2'));
    Rf=Rf+rf;   % mean of covariance of all the trials (1/2)
end

    % step3: Calculate mean of Rf  
Rf=Rf/size(data2,3);   % mean of covariance of all the trials (2/2)


%% step3: Generalized Eigenvalue Decomposition
    [u,v]=eig(Rh,Rf);
    v=diag(v);
    [v,ind] = sort(v,'descend');
    u=u(:,ind);
    w= [u(:,1:mm),u(:,end-mm+1:end)];


end

