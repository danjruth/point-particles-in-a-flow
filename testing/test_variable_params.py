# -*- coding: utf-8 -*-
"""
Created on Mon May  2 15:18:49 2022

@author: danjr
"""

import numpy as np
import matplotlib.pyplot as plt

from pointparticlesinaflow.velocity_fields import gaussian
import pointparticlesinaflow as ppart
from pointparticlesinaflow import equations, EquationOfMotion

import toolkit.parallel

import timeit



class BasicEOM(EquationOfMotion):
    
    additional_pkeys = ['Cm']

    def __init__(self,):
        super().__init__(name='basic_eom',
                         forces=[equations.DragForceSnyder2007(),
                                 #equations.ViscousDragForce(),
                                 #equations.ConstantCLLiftForce(),
                                 equations.PressureForceBubble(),
                                 equations.GravForceBubble()])
        
    def calc_m_eff(self,r):
        return r['Cm']*(r['d']/2)**3*4./3*np.pi
    
L_int = 0.015
u_rms = 0.2
n_modes = 12

n = 5

d = np.ones(n)*0.3e-4
Cm = np.ones_like(d)
Cl = 0.5 * np.ones_like(d)
nu = np.ones_like(d) * 8.9e-7
g = 9.81
g_dir = np.zeros((n,3))
g_dir[:,-1] = -1

v_q = d**2 * g / (18*nu)
v_q_val = v_q[0]

part_params = dict(d=d,Cm=Cm,Cl=Cl,nu=nu,g=g,g_dir=g_dir) # Cd=Cd,
dt = 0.001

vf = gaussian.RandomGaussianVelocityField(n_modes=n_modes,u_rms=u_rms,L_int=L_int)
vf.init_field()

sim_params = dict(t_min=0,t_max=0.01,dt_call=1e-3,n=n,n_call_per_store=1,n_int_per_call=1000)
sim = ppart.Simulation(vf,BasicEOM(),part_params,sim_params.copy())
sim.init_sim()
sim.x[0,:,:] = np.random.uniform(0,vf.L_char*10,size=(n,3))
sim.initialize_to_carrier_velocity()
sim.run(disp=True)
#sim.save_dict()

sim._calc_forces_post()

fig,axs = plt.subplots(1,3,figsize=(9,4),sharex=True,sharey=True)
for i,ax in enumerate(axs):
    ax.plot(sim.t[1:],sim.v[1:,0,i],color='r')
    ax.plot(sim.t[1:],sim.u[1:,0,i],color='b')
    ax.plot(sim.t[1:],sim.v[1:,0,i]-sim.u[1:,0,i],color='k',ls='--')
axs[2].axhline(v_q[0],color='cyan',ls=':')

t_cond = sim.t>1

fig,ax = plt.subplots()
for force in sim.forces:
    ax.hist(sim.forces[force][t_cond,:,2].flatten(),bins=101,density=True,alpha=0.5)
buoyancy_force = sim.forces['grav'][0,0,2]
ax.set_xlim(-buoyancy_force*5,buoyancy_force*5)
ax.axvline(sim.forces['grav'][t_cond,:,2].mean(),color='g')

print(sim.v[t_cond,:,:].mean(axis=(0,1))/v_q_val)

stophere

sim_params['n_int_per_call'] = 5
sim2 = ppart.Simulation(vf,BasicEOM(),part_params,sim_params.copy())
sim2.init_sim()
sim2.x[0,...] = sim.x[0,...]
sim2.v[0,...] = sim.v[0,...]
sim2.run(disp=False)


fig,ax = plt.subplots()
ax.plot(sim.t,sim.v[:,0,0])
ax.plot(sim2.t,sim2.v[:,0,0])

# pl