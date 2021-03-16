% This script is used for randomly generating N=10000 structure parameters 
% The generated columns are [height, gap, period, diameter];

height_range = 30:1:200;
gap_range = 160:1:320;
period_range = 300:1:700;
diameter_range =  80:1:160;
height_l = length(height_range);
gap_l = length(gap_range);
period_l = length(period_range);
diameter_l = length(diameter_range);


sampled = [];
count = 1;
count2 = 1;
N=10000;
height = height_range(1,randi(height_l));
gap = gap_range(1,randi(gap_l));
period = period_range(1,randi(period_l));
diameter = diameter_range(1,randi(diameter_l));
design = [height, gap, period, diameter];
sampled = [sampled; design]; 

while count < N
    count2 = count2+1;
    height = height_range(1,randi(height_l));
    gap = gap_range(1,randi(gap_l));
    period = period_range(1,randi(period_l));
    diameter = diameter_range(1,randi(diameter_l));
    design = [height, gap, period, diameter];
    if ismember(design, sampled,'rows') |  (gap+diameter>=period)
        % remove same parameters and remove parameters that doesn't satisfy
        % the structure requirement: gap+diameter<period
        design;
        continue;
    end
    sampled = [sampled; design];
    count = count+1;
end

save('RCWA_structure.mat','sampled');