%% Auto-weighted Multi-view Graph Learning (AMGL) for data clustering
% Solve the problem: 
%                  min_{F'*F = I} sum_v sqrt(tr(F'*L_v*F)) .
function [result] = AMGL_Clustering(X,cluser_num)
% X: cell array, 1 by view_num, each array is d_v by N.

view_num = size(X,2);
N = size(X{1,1},2);
each_cluster_num = N/cluser_num;
thresh = 10^-8;

% calculate groundtruth
groundtruth = zeros(N,1); % column vector
for c = 1:cluser_num
    for cnt = 1:each_cluster_num
        groundtruth((c-1)*each_cluster_num+cnt,1) = c;
    end
end

% claculate L for all the views
for v = 1:view_num
    fea_v = X{1,v}; 
    W = constructW_PKN(fea_v);
    d = sum(W);
    D = zeros(N);
    for i = 1:N
        D(i,i) = d(i);
    end
%   L(1,v) = {D-W};
    temp_ = diag(sqrt( diag(D).^(-1) ));
    L(1,v) = {eye(N)-temp_*W*temp_};
end 

% do iterations 
maxIter = 100;
alpha = (1/view_num)*ones(1,view_num);
for iter = 1:maxIter
    % Given alpha, update F
    L_sum = zeros(N);
    for v = 1:view_num
        L_sum = L_sum+alpha(v)*L{1,v};
    end
    [eig_vector,eig_value] = myeig(L_sum);
    F = eig_vector(:,1:cluser_num);
    % Given F, update alpha
    for v = 1:view_num
        alpha(v) = 1/(2*sqrt(trace(F'*L{1,v}*F)));
    end
    % calculate objective value
    obj = 0;
    for v = 1:view_num
        obj = obj+sqrt(trace(F'*L{1,v}*F));
    end
    Obj(iter) = obj;
    if iter>2
        Obj_diff = ( Obj(iter-1)-Obj(iter) )/Obj(iter-1);
        if Obj_diff < thresh
            break;
        end
    end
end
%% kmeans discretization
result = ClusteringMeasure(groundtruth,kmeans(F,cluser_num));% ACC NMI Purity









