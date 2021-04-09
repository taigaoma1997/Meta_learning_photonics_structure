%% so far we only need to simulate for models: vae_Full, vae_hybrid, vae_GSNN

model = 'vae_GSNN';  
filename = strcat('data_predicted\param_', model, '_pred.mat');
filename_save = strcat('data_predicted\spectrum\spectrum_param_', model, '_pred.mat');
load(filename);

M = size(param_pred,1);
N = size(param_pred,2);

if isfile(filename_save)==0
    param_pred_re = param_pred.';
    param_pred_re = reshape(param_pred_re, 4,[]);
    param_pred_re = param_pred_re.';

    param_pred_re = param_pred(:,1:4);
    START = 1;
    END = M;
    CURRENT = 1;
end

acc = 10;
stepcase = 5;
show1 = 0;

if CURRENT>END
    fprintf('This simulation is already done! \n');
else
    for i = CURRENT:1:END   % we start from the last data in case the simulation was stopped at the saving part. 
    tic
refls = RCWA_Silicon(param_pred_re(i,1), param_pred_re(i,2), param_pred_re(i,3), param_pred_re(i,4), acc, show1,stepcase);
    spectrum(i,:) = refls;
    CURRENT = i;
    save(filename_save,'spectrum', 'START','END', 'CURRENT');
    i
    toc
    end
end

fprintf('Simulation done! \n');
        
%% Load data 
wave = 380:5:780;
model = 'vae_hybrid';  
filename = strcat('data_predicted\param_', model, '_pred.mat');
filename_save = strcat('data_predicted\spectrum\spectrum_param_', model, '_pred.mat');

load(filename);
load(filename_save);
param_pred_re = param_pred.';
param_pred_re = reshape(param_pred_re, 4,[]);
param_pred_re = param_pred_re.';
param_pred_re = param_pred(:,1:4);

figure(10)
plot(wave, spectrum)
axis([380 780 0 1]);
xlabel('Wavelength/(nm)');
ylabel('Reflection');

%% This part is to transform spectrum data into xyY data and save them

CIE =  importdata('color\cie-cmf.txt');
load('color\D65.mat');

K = D65 * CIE(:,3)/100;   % a normalization constant
temp = transpose(CIE(:,2:4)).*D65/100; 
XYZ = spectrum * transpose(temp)/K;
xyz = XYZ./sum(XYZ, 2);
xyY = xyz;
xyY(:,3) = XYZ(:,2);

%% Save data 
xyY_pred = xyY;
filename_save_xyY = strcat('data_predicted\xyY\xyY_param_', model, '_pred.mat');
save(filename_save_xyY, 'param_test','param_pred','CIE_x','xyY_pred','cie_pred');