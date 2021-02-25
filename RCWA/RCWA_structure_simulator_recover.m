% in case you accidently stopped the simulation and want to reload and
% simulate the rest part. 

load('RCWA_structure1.mat');

load('RCWA_spectrum.mat');
if CURRENT >= END
    fprint('This simulation is done, no need to reload. \n');
end 

s_end = END;
acc = 10;
stepcase = 5;
show1 = 0;

for i = CURRENT:1:s_end  % we start from the last data in case the simulation was stopped at the saving part. 
    refls = RCWA_Silicon(sampled(i,1), sampled(i,2), sampled(i,3), sampled(i,4), acc, show1,stepcase);
    spectrum(i,:) = refls;
    CURRENT = i;
    save('RCWA_spectrum.mat','spectrum', 'START','END', 'CURRENT');
    i
end


fprint('Simulation done! \n');
