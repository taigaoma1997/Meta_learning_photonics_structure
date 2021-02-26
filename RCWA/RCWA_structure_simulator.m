% This is used for simulate the corresponding reflection sppectrum based on
% the structure parameters from the RCWA_structure1.mat

% The parameter columns are [height, gap, period, diameter];


s_start  = 1001; % change this to choose the start simulating point
s_end = 2000; % change this to choose the end simulating point
N = s_end - s_start +1;
START = s_start; 
END = s_end; 
spectrum = [];
load('RCWA_structure1.mat');

acc = 10;
stepcase = 5;
show1 = 0;

for i = s_start:1:s_end
    refls = RCWA_Silicon(sampled(i,1), sampled(i,2), sampled(i,3), sampled(i,4), acc, show1,stepcase);
    spectrum(i,:) = refls;
    CURRENT = i;
    save('RCWA_spectrum.mat','spectrum', 'START','END', 'CURRENT');
    i
end

fprintf('Simulation done! \n');
% it is better to save for 1000 data points everytime. when finish simulating, also change the filename with 0-9  denoting the
% nth thousand datasets and move to the folder \data_generated. eg: RCWA_spectrum_0.mat for 1-1000 data.

