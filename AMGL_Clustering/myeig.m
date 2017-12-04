function [eig_vector,eig_value] = myeig(M)
[vec,val] = eig(M);
if issymmetric(M)
    eig_vector = vec;
    eig_value = val;
else
    temp = diag(val);
    [B,I] = sort(temp,'ascend');
    eig_vector = vec(:,I);
    eig_value = diag(B);
end
    

