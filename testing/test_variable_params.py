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

vf = gaussian.RandomGaussianVelocityField(n_modes=24,u_rms=1,L_int=1)
vf.init_field()


class BasicEOM(EquationOfMotion):
    
    additional_pkeys = ['Cm']

    def __init__(self,):
        super().__init__(name='basic_eom',
                         forces=[equations.ConstantCDDragForce(),
                                 equations.ConstantCLLiftForce(),
                                 equations.PressureForceBubble()])
        
    def calc_m_eff(self,r):
        return r['Cm']*(r['d']/2)**3*4./3*np.pi

        
eom = BasicEOM()

#stophere

n = 200
min_d = 1e-3
max_d = 9e-2
d = np.random.uniform(min_d,max_d,n)
Cd = 0.5 * np.ones_like(d)
Cm = np.ones_like(d)
Cl = 0.5 * np.ones_like(d)

part_params = dict(d=d,Cd=Cd,Cm=Cm,Cl=Cl)
sim_params = dict(t_min=0,t_max=5,dt=0.001,n=n)

sim = ppart.Simulation(vf,eom,part_params,sim_params)
sim.init_sim()

#v = sim._advance(sim.ti)
sim.run(disp=True)

fig,ax = plt.subplots()

def color_d(d):
    return np.array([0.5,0.5,(d-min_d)/(max_d-min_d)])

for i in range(90):
    ax.plot(sim.t,sim.x[:,i,0],color=color_d(d[i]),alpha=0.7)
