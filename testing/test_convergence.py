# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 13:40:18 2021

@author: druth
"""

import point_bubble_JHTDB as pb
from point_bubble_JHTDB import analysis, equations
from point_bubble_JHTDB.velocity_fields import gaussian
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import toolkit.parallel
import time

folder = r'E:\210204_new_pointbubble_data\gaussian_inertial_vs_visc\\'

vf = gaussian.RandomGaussianVelocityField(n_modes=64,u_rms=1,L_int=1)
vf.init_field()

'''
Inertial drag
'''
d,g = analysis.dg_given_nondim(0.1, 0.1, vf.u_char, vf.L_char)
phys_params = {'d':d,
                'g':g,
                'Cm':0.5,
                'Cd':1,
                'Cl':0.0}
# v_q = analysis.quiescent_speed(d, g, 1)
# phys_params = {'d':d,
#                 'g':g,
#                 'Cm':0.5,
#                 'nu':analysis.nu_given_quiescent_visc(d, g, v_q),
#                 'Cl':0.0}
sim_params = {'n_bubs':15,
              'dt':1e-3,
              't_min':0,
              't_max':5,
              'fname':'inertial'}
mr = equations.MaxeyRileyPointBubbleConstantCoefs()

t = pb.TestConvergence(vf,phys_params,sim_params,mr,[1e-3,2e-3,5e-3])
t.run_all()

fig,axs = plt.subplots(1,2,figsize=(9,4))
colors = ['r','b','g']
colors = {dt:colors[i] for i,dt in enumerate(t.complete_sims)}
for dt in t.complete_sims:
    c = t.complete_sims[dt]
    axs[0].plot(c['t'],c['press'][:,:,2],color=colors[dt],alpha=0.5)
    axs[0].plot(np.nan,np.nan,'-',color=colors[dt],label=dt)
    
    x,y = analysis.get_hist(c['press'][...,2].flatten()/c.grav_z,bins=101,)
    axs[1].plot(x,y,color=colors[dt],alpha=0.5)
    
axs[0].legend()