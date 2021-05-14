def RCWA(model):
    
    import scipy
    import matlab.engine
    eng = matlab.engine.start_matlab()
    eng.addpath(eng.genpath('/data/mtobah/Meta_learning_photonics_structure-main_6/RCWA'))
    eng.MODEL_vae(model)
    eng.quit()

    filepath ="./data_predicted/xyY/xyY_param_" + model + "_pred.mat"
    temp = scipy.io.loadmat(filepath)
    return temp