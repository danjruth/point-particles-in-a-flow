# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 11:30:43 2020

@author: ldeike
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import toolkit.comps

folder = toolkit.comps.cf('MILES_C')+r'\\'+r'200717_pointbubble_data\\'
case_name = r'lagrangian_trajecotires_dtFactor0.5'

with open(folder+case_name+'.pkl', 'rb') as handle:
    res = pickle.load(handle)
    
fig,ax = plt.subplots()
ax.plot(res['t'],res['u'].std(axis=1))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(res['x'][:,0,0],res['x'][:,0,1],zs = res['x'][:,0,2])

accel = np.moveaxis(np.diff(res['u'],axis=0),0,-1)/np.diff(res['t'])
accel = np.moveaxis(accel,-1,0)

plt.figure()
plt.plot(res['t'][:-1],accel.std(axis=(1,2)))

plt.figure()
plt.hist(accel[1:2500,:,:].flatten(),density=True,bins=np.linspace(-25,25,1001))