# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 16:55:07 2021

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

vf = gaussian.RandomGaussianVelocityField(n_modes=6,u_rms=1,L_int=1)
vf.init_field()

d,g = analysis.dg_given_nondim(3, 0.1, vf.u_char, vf.L_char)
phys_params = {'d':d,
                'g':g,
                'Cm':0.5,
                'Cd':1,
                'Cl':0.0}
sim_params = {'n_bubs':5,
              'dt':1e-3,
              't_min':0,
              't_max':0.5,
              'fname':'inertial'}
mr2 = equations.MaxeyRileyPointBubbleConstantCoefs()
sim = pb.Simulation(vf,phys_params,sim_params,mr2)
sim.init_sim(g_dir='random')
t1 = time.time()
sim.run()
print(time.time()-t1)

phys_params = {'d':d,
                'g':g,
                'Cm':0.5,
                'nu':analysis.nu_given_quiescent_visc(d,g,sim.v_q),
                'Cl':0.0}
mr = equations.MaxeyRileyPointBubbleConstantCoefsVisc()
sim2 = pb.Simulation(vf,phys_params,sim_params,mr)
sim2.init_sim(g_dir='random')
sim2.x[0,...] = sim.x[0,...].copy()
sim2.g_dir = sim.g_dir.copy()
t1 = time.time()
sim2.run()
print(time.time()-t1)

fig,ax = plt.subplots()
ax.plot(sim.t,sim.x[:,0,:],ls='-')
ax.plot(sim2.t,sim2.x[:,0,:],ls='--')