%%
wave = 380:5:780;
model = 'inn';  % change this to other model: inn, vae, gan, inverse, tandem 
filename = strcat('data_predicted\param_', model, '_pred.mat');
filename_save = strcat('data_predicted\spectrum\spectrum_param_', model, '_pred.mat');
load(filename);
load(filename_save);
param_pred_re = reshape(param_pred, [], 4);

figure(1)
plot(wave, spectrum)
axis([380 780 0 1]);
xlabel('Wavelength/(nm)');
ylabel('Reflection');

%%  only run this for inn model

wrong = [];
M = size(spectrum,1);
N = size(spectrum, 2);

for i = 1:1:M
    d_s_1 = spectrum(i, 2:N) - spectrum(i,1:(N-1));
    if (max(d_s_1)>=0.25)
        wrong = [wrong, i];
    end
end

wrong_spectrum = [];
N = size(wrong, 2);
acc = 10;
stepcase = 5;
show1 = 0;
figure(2)
for j = 1:1:N
    i = wrong(j)
    wrong_spectrum(j,:) =  RCWA_Silicon(param_pred_re(i,1), param_pred_re(i,2), param_pred_re(i,3), param_pred_re(i,4), acc, show1,stepcase);
   
    subplot(4, 10,j)
    plot(wave, spectrum(i,:), wave, wrong_spectrum(j,:))
    axis([380 780 0 1]);
    xlabel('Wavelength/(nm)');
    ylabel('Reflection'); 
    spectrum(i,:) = wrong_spectrum(j,:);
end
    
save(filename_save,'spectrum', 'START','END', 'CURRENT');
%% This part is to transform spectrum data into xyY data and save those data

CIE =  importdata('color\cie-cmf.txt');
load('color\D65.mat');

K = D65 * CIE(:,3)/100;   % a normalization constant
temp = transpose(CIE(:,2:4)).*D65/100; 
XYZ = spectrum * transpose(temp)/K;
xyz = XYZ./sum(XYZ, 2);
xyY = xyz;
xyY(:,3) = XYZ(:,2);

CELL = {'inn','gan','vae'};
if any(strcmp(CELL,model))
    xyY_pred =  reshape(xyY, [], 15); 
else
    xyY_pred = xyY;
end

filename_save_xyY = strcat('data_predicted\xyY\xyY_param_', model, '_pred.mat');
save(filename_save_xyY, 'param_test','param_pred','CIE_x','xyY_pred');
%%