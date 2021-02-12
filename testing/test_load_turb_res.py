# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 09:53:48 2021

@author: druth
"""

import numpy as np
import matplotlib.pyplot as plt
from point_bubble_JHTDB import classes, analysis, equations
from point_bubble_JHTDB.velocity_fields import jhtdb

vf = jhtdb.JHTDBVelocityField()
vf.init_field()

mr = equations.MaxeyRileyPointBubbleConstantCoefs()
              
# create the simulation
phys_params = {}
sim_params = {}
sim = classes.Simulation(vf,phys_params,sim_params,mr)
sim.add_data(r'E:\210204_new_pointbubble_data\Fr1_dstar0.12.pkl')

a = analysis.CompleteSim(sim,rotated=True)

fig,ax = plt.subplots()
ax.plot(a['t']/a.T_vf,a['v'][...,2].mean(axis=1)/a.v_q,)

fig,ax = plt.subplots()
ax.plot(a['t']/a.T_vf,a['drag'][...,2].mean(axis=1)/a.grav_z,)
ax.plot(a['t']/a.T_vf,a['press'][...,2].mean(axis=1)/a.grav_z,)
ax.plot(a['t']/a.T_vf,a['lift'][...,2].mean(axis=1)/a.grav_z,)

fig,ax = plt.subplots()
ax.plot(a['t']/a.T_vf,a['drag'][...,2].std(axis=1)/a.grav_z,)
ax.plot(a['t']/a.T_vf,a['press'][...,2].std(axis=1)/a.grav_z,)
ax.plot(a['t']/a.T_vf,a['lift'][...,2].std(axis=1)/a.grav_z,)

fig,ax = plt.subplots()
i = 5
ax.plot(a['t']/a.T_vf,a['drag'][:,i,2]/a.grav_z,)
ax.plot(a['t']/a.T_vf,a['press'][:,i,2]/a.grav_z,)
ax.plot(a['t']/a.T_vf,a['lift'][:,i,2]/a.grav_z,)