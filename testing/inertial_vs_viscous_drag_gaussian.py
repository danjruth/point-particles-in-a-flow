# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 10:53:31 2021

@author: druth
"""

from point_bubble_JHTDB import model, analysis
from point_bubble_JHTDB.velocity_fields import gaussian
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import toolkit.parallel
import time

folder = r'E:\210204_new_pointbubble_data\\'

# create the velocity field


dstar = 0.1
Fr_vec = np.concatenate([np.geomspace(0.1,6,10,endpoint=False),np.linspace(6,10,4)])

res = []

for Fr in Fr_vec:
    
    def do_job(_):
    
        vf = gaussian.RandomGaussianVelocityField(n_modes=24,u_rms=1,L_int=1)
        vf.init_field()
        
        '''
        Inertial drag
        '''
        d,g = model.phys_params_given_nondim(Fr, dstar, vf.u_char, vf.L_char)
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
        mr = model.MaxeyRileyPointBubbleConstantCoefs()
        sim_inertial = model.Simulation(vf,phys_params,sim_params,mr)
        sim_inertial.init_sim(g_dir='random')
        sim_inertial.run()
        a = analysis.CompleteSim(sim_inertial)
        a_mean = a['v'][2000:,:,2].mean()
        a_v_q = a.v_q
        del a
        
        '''
        Viscous drag
        '''
        phys_params = {'d':sim_inertial.d,
                        'g':sim_inertial.g,
                        'Cm':0.5,
                        'Cd':np.nan,
                        'Cl':0.0,
                        'nu':model.nu_given_quiescent_visc(sim_inertial.d,sim_inertial.g,sim_inertial.v_q)}
        mrv = model.MaxeyRileyPointBubbleConstantCoefsVisc()
        sim_visc = model.Simulation(vf,phys_params,sim_params,mrv)
        sim_visc.init_sim()
        sim_visc.g_dir = sim_inertial.g_dir
        sim_visc.x[0,:,:] = sim_inertial.x[0,:,:]
        sim_visc.run()
        b = analysis.CompleteSim(sim_visc)
        
        return [a_mean/a_v_q,b['v'][2000:,:,2].mean()/a_v_q]
    
    res_Fr = toolkit.parallel.parallelize_job(do_job,range(36))
    res.append(res_Fr)
    
res = np.array(res)
res_mean = np.mean(res,axis=1)
fig,ax = plt.subplots()
ax.plot(Fr_vec,res_mean[:,0],'x-',label='non-linear drag, $C_\mathrm{D}=1$')
ax.plot(Fr_vec,res_mean[:,1],'x-',label='linear drag, $C_\mathrm{D}=24/\mathrm{Re}$')
ax.set_xlabel(r'''$\mathrm{Fr} = u'/\sqrt{gd}$''')
ax.set_ylabel(r'$\langle v_z \rangle / v_\mathrm{q}$')
ax.legend()
fig.tight_layout()