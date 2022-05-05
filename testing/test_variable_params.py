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
                         forces=[#equations.DragForceSnyder2007(),
                                 equations.ConstantCDDragForce(),
                                 #equations.ConstantCLLiftForce(),
                                 equations.PressureForceBubble(),
                                 equations.GravForceBubble()])
        
    def calc_m_eff(self,r):
        return r['Cm']*(r['d']/2)**3*4./3*np.pi
    
L_int = 1
T_int = 1
u_rms = 1
n_modes = 12

n = 1

d = np.ones(n)*3e-4
Cd = 0.5 * np.ones_like(d)
Cm = np.ones_like(d)
Cl = 0.5 * np.ones_like(d)
nu = np.ones_like(d) * 8.9e-7
g = 9.81
g_dir = np.zeros((n,3))
g_dir[:,-1] = -1

part_params = dict(d=d,Cd=Cd,Cm=Cm,Cl=Cl,nu=nu,g=g,g_dir=g_dir)
dt = 0.0001

vf = gaussian.RandomGaussianVelocityField(n_modes=n_modes,u_rms=u_rms,L_int=L_int)
vf.init_field()

sim_params = dict(t_min=0,t_max=0.1,dt=dt,n=n,n_call_per_timestep=5)

sim = ppart.Simulation(vf,BasicEOM(),part_params,sim_params)
sim.init_sim()
sim.initialize_to_carrier_velocity()
sim.run(disp=False,save_every=50)
sim.save_dict()

vf2 = gaussian.RandomGaussianVelocityField(n_modes=n_modes,u_rms=u_rms,L_int=L_int)
vf2.init_field()
sim2 = ppart.Simulation(vf2,BasicEOM(),part_params,sim_params)
sim2.init_sim()
sim2.from_dict()
