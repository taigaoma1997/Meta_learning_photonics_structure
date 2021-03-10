
% so far we only need to simulate for models: vae, inn, inverse, tandem. 
model = 'tandem';  
filename = strcat('data_predicted\param_', model, '_pred.mat');
filename_save = strcat('data_predicted\spectrum\spectrum_param_', model, '_pred.mat');

load(filename);
M = size(param_pred,1);
N = size(param_pred,2);
param_pred_re = reshape(param_pred, [], 4);

START = 1;
END = size(param_pred_re,1);

CURRENT = 1;

acc = 10;
stepcase = 5;
show1 = 0;

spectrum= [];
for i = START:1:END   % we start from the last data in case the simulation was stopped at the saving part. 
    refls = RCWA_Silicon(param_pred_re(i,1), param_pred_re(i,2), param_pred_re(i,3), param_pred_re(i,4), acc, show1,stepcase);
    spectrum(i,:) = refls;
    CURRENT = i;
    save(filename_save,'spectrum', 'START','END', 'CURRENT');
    i
end

CURRENT = CURRENT +1;

fprintf('Simulation done! \n');