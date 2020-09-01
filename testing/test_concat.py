# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 15:51:01 2020

@author: ldeike
"""

from point_bubble_JHTDB import analysis
import pickle
import toolkit.comps
import numpy as np
import matplotlib.pyplot as plt

folders = [toolkit.comps.cf('MILES_D')+r'200809_pointbubble_data\\',
           toolkit.comps.cf('MILES_D')+r'200809_pointbubble_data\\',]

case_names = [r'res_beta0.75_A0.010_Cm0.50_Cl0.00_Cd0.50_pressureTerm1.000_liftTerm1.000',
              r'res_beta0.75_A0.010_Cm0.50_Cl0.00_Cd0.50_pressureTerm1.000_liftTerm1.000_long']

ds = [pickle.load(open(folder+case_name+'.pkl','rb')) for (case_name,folder) in zip(case_names,folders)]

stophere

res2 = analysis.concat_cases(ds)
stophere
#c = analysis.load_case(res2) # do not save res2 after calling this -- velgrad key is deleted

pickle.dump(res2,open(folders[1]+case_names[0]+'.pkl','wb'))

#plt.figure()
#plt.pcolormesh(c['v'][:,:,2])