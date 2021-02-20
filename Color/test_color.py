# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 21:39:23 2021

@author: mtg
"""

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat

#from color_sys import ColourSystem

refls = scipy.io.loadmat('refls_paper.mat')
spec = np.array(list(refls.items())[3][1])


def spec_to_xyY(spec):
    # calculate the xyY from a spectrum. spec should be  from 380-780nm with 5nm range, and array with size(1,81)
    cmf_path = 'cie-cmf.txt'
    cmf = np.loadtxt(cmf_path, usecols=(1,2,3))
    SD65 = scipy.io.loadmat('D65.mat')
    SD65 = np.array(list(SD65.items())[3][1])/100
    #SD65 = SD65/SD65
    K = np.dot(SD65,cmf)[0,1]
    XYZ = np.dot(spec*SD65, cmf)/K
    
    #b=np.array([0.3127, 0.3291, 0.3582])
    
    den = np.sum(XYZ, axis=1)
    xyz=XYZ.copy()
    for i in range(0,16):
        xyz[i,:]=XYZ[i,:]/den[i]
        
    xyY = xyz.copy()
    xyY[:,2] = XYZ[:,1]
    
    return xyY

a =spec_to_xyY(spec)
    