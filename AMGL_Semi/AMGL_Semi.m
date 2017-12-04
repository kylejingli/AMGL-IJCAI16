%% Auto-weighted Multi-view Graph Learning (AMGL) for semi-supervised classification
% Solve the problem;
%                   min_{f_i = y_i, 1 <= i <= l} sum_v sqrt(trance(F'*L_v*F))
function [result] = AMGL_Semi(X,class_num,ratio)
% X: cell array, 1 by view_num, each array is d_v by N
% class_num: the number of class
% ratio: the propotion of labeled data, e.g., 0.1
% some default parameters: thresh = 10^-8; maxIter = 100;  

view_num = size(X,2);
N = size(X{1,1},2);
each_class_num = N/class_num;
thresh = 10^-8;
part = floor(ratio*each_class_num); % Each class have the same size of data



labeled_N = part*class_num;
list = sort(randperm(each_class_num,part));  % Random select the labeled data!!!
List = [];
for c = 1:class_num
    List = [List list+(c-1)*each_class_num];
end
List_ = setdiff(1:1:N,List); % the No. of unlabeled data
    

samp_label = zeros(N,class_num); % column vector
for c = 1:class_num
    samp_label((c-1)*each_class_num+(1:each_class_num),c) = ones(each_class_num,1);
end

groundtruth = zeros(N,class_num);
groundtruth(1:labeled_N,:) = samp_label(List,:);
groundtruth((labeled_N+1):N,:) = samp_label(List_,:);

F_l = groundtruth(1:labeled_N,:);

% Construct the affinity matrix for each view data
for v = 1:view_num
    temp = X{1,v};
    [row_num,col_num] = size(temp);
    fea_v = zeros(row_num,col_num);
    fea_v(:,1:labeled_N) = temp(:,List);
    fea_v(:,(labeled_N+1):N) = temp(:,List_); 

    W = constructW_PKN(fea_v); % fea_v is a d_i by n matrix
    d = sum(W);       
    D = diag(d);
    temp_ = diag(sqrt( diag(D).^(-1) ));
    L(1,v) = { eye(N)-temp_*W*temp_ };
end  

% Iterately solve the target problem
maxIter = 100;
alpha = 1/view_num*ones(1,view_num);

for iter = 1:maxIter
    % Given alpha, update F_u
    L_sum = zeros(N);
    for v = 1:view_num
        L_sum = L_sum+alpha(v)*L{1,v};
    end
    L_ul = L_sum((labeled_N+1):N,1:labeled_N);
    L_uu = L_sum((labeled_N+1):N,(labeled_N+1):N);
    F_u = -0.5*inv(L_uu)*L_ul*F_l;
    % Given F_u, update alpha
    F = [F_l;F_u];
    for v = 1:view_num
          alpha(v) = 0.5/sqrt(trace(F'*L{1,v}*F));
    end
    % Calculate objective value
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
    
cnt = 0;
for u = (labeled_N+1):N
    pos = find(F(u,:) == max(F(u,:)));
    y = zeros(1,class_num);
    y(1,pos) = 1;
    if y == groundtruth(u,:);
       cnt = cnt+1;
    end
end

result = cnt/(N-labeled_N);

isdescend = isequal(Obj,sort(Obj,'descend'));

