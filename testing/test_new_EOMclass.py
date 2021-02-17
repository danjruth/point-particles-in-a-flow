# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 16:55:07 2021

@author: druth
"""



import point_bubble_JHTDB as pb
#stophere

from point_bubble_JHTDB import analysis, equations
from point_bubble_JHTDB.velocity_fields import gaussian, two_dimensional
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import toolkit.parallel
import time

vf = gaussian.RandomGaussianVelocityField(n_modes=64,u_rms=1,L_int=1)
#vf = two_dimensional.SteadyPlanarPoiseuilleFlow()
vf.init_field()

d,g = analysis.dg_given_nondim(4, 0.1, vf.u_char, vf.L_char)

phys_params = {'d':d,
                'g':g,
                'Cm':0.5,
                'Cd':1,
                'Cl':0.5}

sim_params = {'n_bubs':500,
              'dt':1e-3,
              't_min':0,
              't_max':18,
              'fname':'inertial'}

'''
Snyder, quiescent field
'''
phys_params = {'d':d,
                'g':g,
                'Cm':0.5,
                'nu':analysis.nu_given_Req(d, g, 100, 10), # pick nu given the Reynolds number that corresponds to a given velocity
                'Cl':0.0}
eom = equations.MaxeyRileyPointBubbleSnyder2007()
vf_q = pb.VelocityField()
sim_params_q = sim_params.copy()
sim_params_q['n_bubs'] = 1
simq = pb.Simulation(vf_q,phys_params,sim_params_q,eom)
simq.init_sim(g_dir='random')
simq.v[0] = np.array([0.001,0.001,0.001])
t1 = time.time()
simq.run()
print(time.time()-t1)
a_q = analysis.CompleteSim(simq,rotated=True)
a_q['v'].mean(axis=(0,1))
v_q = np.mean(a_q['v'][-1,:,2])

plt.figure()
plt.plot(a_q['t'],a_q['v'][:,0,:])



'''
Snyder
'''

eom = equations.MaxeyRileyPointBubbleSnyder2007()
sim2 = pb.Simulation(vf,phys_params,sim_params,eom)
sim2.init_sim(g_dir='-z',pos_lims=((0,0,0),(1000,1000,1000)))
t1 = time.time()
sim2.run()
print(time.time()-t1)
a2 = analysis.CompleteSim(sim2,rotated=False)
print('Snyder: ',str(a2['v'].mean(axis=(0,1))/v_q))

'''
Constant Cd, matching mean Cd from Snyder
'''

Re = np.linalg.norm((a2['v']-a2['u']),axis=-1) * a2.d / a2.nu
Cd = equations.calc_Cd_Snyder(Re)
Cd_mean = Cd.mean()
fig,ax = plt.subplots()
ax.hist(Cd.flatten(),bins=np.geomspace(0.01,10,101),cumulative=False,density=True,alpha=0.5)
ax.hist(Re.flatten(),bins=np.geomspace(0.1,1000,101),cumulative=False,density=True,alpha=0.5)


phys_params = {'d':d,
                'g':g,
                'Cm':0.5,
                'Cd':Cd_mean,
                'Cl':0.5}
eom = equations.MaxeyRileyPointBubbleConstantCoefs()
sim3 = pb.Simulation(vf,phys_params,sim_params,eom)
sim3.init_sim(g_dir='-z')
sim3.g_dir = sim2.g_dir.copy()
sim3.x[0] = sim2.x[0].copy()
t1 = time.time()
sim3.run()
print(time.time()-t1)
a3 = analysis.CompleteSim(sim3,rotated=False)
print('Matching mean Cd from Snyder: ',str(a3['v'].mean(axis=(0,1))/v_q))

plt.figure()
plt.plot(a2['t'],a2['v'][:,0,:])
plt.plot(a3['t'],a3['v'][:,0,:],ls='--')

curv = analysis.get_curvature(a3['v'],a3['t'])
x,y = analysis.get_hist(curv.flatten(),bins=np.geomspace(0.001,1000,41))
plt.figure()
plt.semilogx(x,analysis.get_powerlaw(x,y,roll_window=1))

# '''
# Snyder, quiescent field
# '''

# eom = equations.MaxeyRileyPointBubbleSnyder2007()
# vf_q = pb.VelocityField()
# sim3 = pb.Simulation(vf_q,phys_params,sim_params,eom)
# sim3.init_sim(g_dir='random')
# sim3.v[0] = np.array([0.001,0.001,0.001])
# t1 = time.time()
# sim3.run()
# print(time.time()-t1)
# a3 = analysis.CompleteSim(sim3,rotated=True)
# a3['v'].mean(axis=(0,1))

# plt.figure()
# plt.plot(a3['t'],a3['v'][:,0,:])