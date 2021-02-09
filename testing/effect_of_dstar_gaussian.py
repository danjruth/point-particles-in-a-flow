# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 14:39:17 2021

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

folder = r'E:\210204_new_pointbubble_data\\'

# create the velocity field


Fr = 1
dstar_vec = np.geomspace(0.1,2,4)
res = []

for dstar in dstar_vec:
    
    def do_job(_):
    
        vf = gaussian.RandomGaussianVelocityField(n_modes=24,u_rms=1,L_int=1)
        vf.init_field()
        
        '''
        Inertial drag
        '''
        d,g = equations.phys_params_given_nondim(Fr, dstar, vf.u_char, vf.L_char)
        phys_params = {'d':d,
                        'g':g,
                        'Cm':0.5,
                        'Cd':1,
                        'Cl':0.0}
        sim_params = {'n_bubs':100,
                      'dt':1e-3,
                      't_min':0,
                      't_max':6,
                      'fname':'inertial'}
        mr = pb.MaxeyRileyPointBubbleConstantCoefs()
        sim_inertial = pb.Simulation(vf,phys_params,sim_params,mr)
        sim_inertial.init_sim(g_dir='random')
        sim_inertial.run()
        a = analysis.CompleteSim(sim_inertial)
        a_mean = a['v'][2000:,:,2].mean()
        a_v_q = a.v_q
        del a
        del sim_inertial
        
        return a_mean/a_v_q
    
    res.append(toolkit.parallel.parallelize_job(do_job,range(18)))
    
fig,ax = plt.subplots()
ax.semilogx(dstar_vec,np.mean(res,axis=1),'x-')

vf = gaussian.RandomGaussianVelocityField(n_modes=24,u_rms=1,L_int=1)
vf.init_field()

d,g = equations.phys_params_given_nondim(3, 0.1, vf.u_char, vf.L_char)
phys_params = {'d':d,
                'g':g,
                'Cm':0.5,
                'Cd':1,
                'Cl':0.0}
sim_params = {'n_bubs':500,
              'dt':1e-3,
              't_min':0,
              't_max':6,
              'fname':'inertial'}
mr = pb.MaxeyRileyPointBubbleConstantCoefs()
sim_inertial = pb.Simulation(vf,phys_params,sim_params,mr)
sim_inertial.init_sim(g_dir='random')
sim_inertial.run()
a = analysis.CompleteSim(sim_inertial)
a_mean = a['v'][2000:,:,2].mean(axis=0)