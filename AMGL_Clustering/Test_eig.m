clear;
M = rand(5);
M_sys = (M+M')/2;
[vec,val] = myeig(M);
[vec_,val_] = myeig(M_sys);