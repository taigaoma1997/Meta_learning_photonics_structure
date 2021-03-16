import matlab.engine
import numpy as np
import scipy.io
eng = matlab.engine.start_matlab() #Load matlab
eng.addpath(eng.genpath('/Users/mustafa/Downloads/Meta_learning_photonics_structure-main 2/RCWA')) #Adding path to simulation function
filepath = './data_predicted/param_vae_pred.mat' #File path to predicted structures

#Set up data

temp = scipy.io.loadmat(filepath) #load matlab data as dict
key = list(temp.keys())[0] #grab the key for the dict.
data = temp[key] #load the data as an array
spectrum = [] 

#Values that are sent to matlab need to be sent as floats

acc = 10.
show1 = 0.
stepcase = 5.
for i in range(len(data)):
    struct = [float(data[i][0]),float(data[i][1]),float(data[i][2]),float(data[i][3])] #load a single structure
    print('structure ' + str(i+1))
    ret = eng.RCWA_Silicon(struct[0],struct[1],struct[2],struct[3],acc,show1,stepcase) #Runs the RCWA simulation through matlab
    spectrum.append(ret)
print('Simulation Complete!')