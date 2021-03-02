% this combines all the simulated spectrum data together and transform the
% spec to xyY data and select the qualified data where period>gap+diameter

%%  This part is used to load all 10000 data together

load('RCWA_structure1.mat');
RCWA_spectrum = [];
RCWA_xyY = [];
files = dir('data_generated\**');

for i = 1:1:9
    filename = files(2+i,1).name;
    fullname = fullfile('data_generated',filename);
    load(fullname);
    if END>CURRENT
        fprintf('imperfect data');
    end
    RCWA_spectrum(START:END,:) = spectrum(START:END, :);
end 
data_rcwa_old = [sampled, RCWA_spectrum];

%% transfrom spectrum to xyY 

CIE =  importdata('color\cie-cmf.txt');
load('color\D65.mat');

K = D65 * CIE(:,3)/100;   % a normalization constant
temp = transpose(CIE(:,2:4)).*D65/100; 
XYZ = RCWA_spectrum * transpose(temp)/K;
xyz = XYZ./sum(XYZ, 2);
xyY = xyz;
xyY(:,3) = XYZ(:,2);

data_rcwa_xyY_old = [sampled, xyY];

%% get ride of some unqualified data where period < diameter+gap and save final data, which contains 8411 spectrum information.

data_rcwa = [];
data_rcwa_xyY = [];
for i = 1:1:10000
    if ((sampled(i,2)+sampled(i,4))<sampled(i,3))
        data_rcwa = [data_rcwa; data_rcwa_old(i,:)];
        data_rcwa_xyY = [data_rcwa_xyY; data_rcwa_xyY_old(i,:)];
    end
end

save('data_generated\RCWA_spectrum_all.mat','data_rcwa');
save('data_generated\RCWA_xyY_all.mat','data_rcwa_xyY');   % data is ready for training.

%%

        




