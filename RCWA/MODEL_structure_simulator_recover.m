% so far we only need to simulate for models: vae, inn, inverse, tandem. 
model = 'tandem';
filename = strcat('data_predicted\param_', model, '_pred.mat');
filename_save = strcat('data_predicted\spectrum\spectrum_param_', model, '_pred.mat');

load(filename);
load(filename_save);
param_pred_re = reshape(param_pred, [], 4);

acc = 10;
stepcase = 5;
show1 = 0;

if CURRENT>END
    fprintf('This simulation is already done! \n');
else
    for i = CURRENT:1:END   % we start from the last data in case the simulation was stopped at the saving part. 
    refls = RCWA_Silicon(param_pred_re(i,1), param_pred_re(i,2), param_pred_re(i,3), param_pred_re(i,4), acc, show1,stepcase);
    spectrum(i,:) = refls;
    CURRENT = i;
    save(filename_save,'spectrum', 'START','END', 'CURRENT');
    i
    end
end

fprintf('Simulation done! \n');