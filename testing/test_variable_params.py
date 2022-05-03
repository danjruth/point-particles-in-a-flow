# -*- coding: utf-8 -*-
"""
Created on Mon May  2 15:18:49 2022

@author: danjr
"""

import numpy as np
#import matplotlib.pyplot as plt

from pointparticlesinaflow.velocity_fields import gaussian
import pointparticlesinaflow as ppart
from pointparticlesinaflow import equations, EquationOfMotion

vf = gaussian.RandomGaussianVelocityField(n_modes=12,u_rms=1,L_int=1)
vf.init_field()

drag = equations.ConstantCDDragForce()

class BasicEOM(EquationOfMotion):

    def __init__(self,):
        super().__init__(name='just_drag',
                         forces=[drag])
        
    def calc_m_eff(self,r):
        return r['Cm']*(r['d']/2)**3*4./3*np.pi
        
eom = BasicEOM()


n = 20
d = np.random.uniform(1e-3,2e-3,n)
Cd = 0.5 * np.ones_like(d)
Cm = np.ones_like(d)

part_params = dict(d=d,Cd=Cd,Cm=Cm)
sim_params = dict(t_min=0,t_max=2,dt=0.01,n=n)

sim = ppart.Simulation(vf,eom,part_params,sim_params)
sim.init_sim()

sim._advance(sim.ti)