load('data_predicted\param_inverse_pred.mat');
%load('testing_data.mat');

m = size(param_pred, 1);
refls = zeros(m, 81);
acc=10;
show1=0;
for i =1:1:m
    refls(i,:) =RCWA_Silicon(param_pred(i,1),param_pred(i,2),param_pred(i,3),param_pred(i,4),acc, show1);
    i
end

save('data_predicted\spectrum_inverse_pred.mat','refls')