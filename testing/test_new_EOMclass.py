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
              't_max':0.2,
              'fname':'inertial'}
mr2 = equations.MaxeyRileyPointBubbleConstantCoefs2()
sim = pb.Simulation(vf,phys_params,sim_params,mr2)
sim.init_sim(g_dir='random')
t1 = time.time()
sim.run()
print(time.time()-t1)

mr = equations.MaxeyRileyPointBubbleConstantCoefs()
simold = pb.Simulation(vf,phys_params,sim_params,mr)
simold.init_sim(g_dir='random')
simold.x[0,...] = sim.x[0,...].copy()
simold.g_dir = sim.g_dir.copy()
t1 = time.time()
simold.run()
print(time.time()-t1)

fig,ax = plt.subplots()
ax.plot(sim.t,sim.x[:,0,:],ls='-')
ax.plot(simold.t,simold.x[:,0,:],ls='--')