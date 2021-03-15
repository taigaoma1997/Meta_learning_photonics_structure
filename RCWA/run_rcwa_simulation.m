function refls= RCWA_Silicon(period, height, diameter, acc, stepcase, structure, material, plot_fig)
    wave = 380:10:780;
    refls =[];
    medium=0;
    
    parfor i = 1:length(wave)
        wavelength=wave(i);
        addpath('./RCWA/RETICOLO V8/reticolo_allege')

        [prv,vmax]=retio([],inf*1i); % never write on the disc (nod to do retio)
        
        material_data = readmatrix(sprintf('./RCWA/material_database/%s.csv', material));
        wl = material_data(:, 1) * 1e3;
        R = material_data(:, 2);
        I = material_data(:, 3);
        n = interp1(wl, R, wavelength)+1i*interp1(wl, I, wavelength);

        periods = [period,period];% same unit as wavelength
        n_air = 1;% refractive index of the top layer
        n_glass = 1.5;% refractive index of the bottom layer
        angle_theta = 0;
        k_parallel = n_air*sin(angle_theta*pi/180);
        angle_delta = 0;
        parm = res0; % default parameters for "parm"
        parm.sym.pol = 1; % TE
        parm.res1.champ = 1; % the eletromagnetic field is calculated accurately
        parm.sym.x=0;parm.sym.y=0;% use of symetry
        parm.res1.trace = 0; % do not plot the index distribution

        nn=[acc,acc];
        
        if structure == 'rod'
            textures = cell(1,3);
            textures{1}= n_air; % uniform texture
            textures{2}={n_air,[0,0,diameter,diameter,n,stepcase]};
            textures{3}= n; % uniform texture
            profile={[200, height, 10000],[1,2,3]}; % how many layers and its corresponding refractive index

            aa=res1(wavelength,periods,textures,nn,k_parallel,angle_delta,parm);
            two_D=res2(aa,profile);
            n_order = size(two_D.TEinc_top_transmitted.efficiency_TE,1);
            refls(i) = sum(two_D.TEinc_top_reflected.efficiency_TE);
        end
    end

    if plot_fig
        plot(wave, refls)
        xlabel('Wavelength/(nm)');
        ylabel('Efficiency');
        legend({'R'})
        saveas(gcf, 'res.png')
    end

    res = transpose([wave; refls]);
    writematrix(res, sprintf('./meta_learning_data/%s_p%d_h%d_d%d.csv', material, period, height, diameter));
end