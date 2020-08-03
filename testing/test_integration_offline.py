# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 12:04:35 2020

@author: ldeike
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import toolkit.comps
import pandas as pd
from point_bubble_JHTDB import analysis, model

case_name = r'res_beta0.50_A0.042_Cm0.50_Cl0.25_Cd0.80_pressureTerm1.000_liftTerm1.000'
res = analysis.load_case(toolkit.comps.cf('MILES_C')+r'\\'+r'200717_pointbubble_data\\'+case_name+r'.pkl')

accel = np.moveaxis(np.diff(res['v'],axis=0),0,-1) / np.diff(res['t'])
accel = np.moveaxis(accel,-1,0)

grav = np.zeros_like(res['press'])
grav[:,:,2] = res['grav_z']
sum_of_forces = res['press']+res['drag']+res['lift']+grav

fig,ax = plt.subplots()
ax.scatter(accel[:,:,2].flatten(),sum_of_forces[:-1,:,2].flatten())