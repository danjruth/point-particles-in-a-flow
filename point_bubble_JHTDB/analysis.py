# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 11:55:39 2020

@author: ldeike
"""

import numpy as np
import pickle
from . import model
import pandas as pd

class CompleteSim():
    
    def __init__(self,sim,norm=False):
        
        # should arrays be normalized when returned
        self.norm = norm
        
        # copy parameters from the Simulation
        self.velocity_field = sim.velocity_field
        self.phys_params = sim.phys_params
        self.sim_params = sim.sim_params
        self.eom = sim.eom
        self = model.assign_attributes(self,self.phys_params,self.sim_params)
        
        # characteristic scales of the velocity field
        self.u_vf = sim.velocity_field.u_char
        self.L_vf = sim.velocity_field.L_char
        self.T_vf = sim.velocity_field.T_char
            
        # nondimensional numbers
        self.dstar = self.d / self.L_vf
        self.dstar_by_Cd = self.dstar / self.Cd
        self.beta = self.u_vf / self.v_q
        self.Fr = self.u_vf / np.sqrt(self.d*self.g)
        
        # get the forces and rotate everything so z is aligned with gravity for each bubble
        self._forces_and_rotation(sim)
        
        # minimum index from which to return the data through __call__
        self.ti_min = 0
        
    def set_min_valid_time(self,n,units='T_vf'):
        '''Set the index beyond which to return data, corresponding to n*units
        '''
        unit_durs = {'T_vf':self.T_vf,
                     'data':1,
                     'vq_by_g':self.v_q/self.g,
                     'u_vf_by_g':self.u_vf/self.g}
        self.ti_min = int(round(n*unit_durs[units]))
        
    def _forces_and_rotation(self,sim):
        '''Compute the forces acting on the particle, and rotate all vectors to
        the gravity-aligned coordinate system
        '''
        
        # slip, vorticity, and bubble volume
        slip = sim.v[:-1] - sim.u[1:]
        vort = get_vorticity(sim.velgrad)
        vol = (sim.d/2.)**3 * 4./3 * np.pi
        
        # gravity force
        grav_z = sim.g * vol
        
        # calculate the forces, in DNS coords initially
        press = []
        drag = []
        lift = []
        u_times_deldotu_all = []
        for i in [0,1,2]:
            
            u_times_deldotu = np.sum(sim.velgrad[1:,:,i,:]*sim.u[1:],axis=-1)
            u_times_deldotu_all.append(u_times_deldotu)
            press.append((1+sim.Cm)* vol * (sim.dudt[1:,:,i] + u_times_deldotu))
            drag.append(-1*sim.Cd * 0.5 * np.pi * (sim.d/2)**2 * slip[...,i] * np.linalg.norm(slip[:,...],axis=-1))        
            lift.append(-1 * sim.Cl * np.cross(slip,vort[1:])[...,i] * vol)
            
        # make each arrays
        sim.u_times_deldotu = np.moveaxis(np.array(u_times_deldotu_all),0,-1)
        sim.press = np.moveaxis(np.array(press),0,-1)
        sim.drag = np.moveaxis(np.array(drag),0,-1)
        sim.lift = np.moveaxis(np.array(lift),0,-1)
        sim.slip = slip # sim.v - sim.u # differs from the definition of slip used to calculate the forces!
        sim.vort = vort
        sim.grav_z = grav_z
        
        # store rotated numerical results in a dict
        r = {}
        fields_rot = ['v','u','slip','x','vort','dudt','u_times_deldotu','press','drag','lift']
        for f in fields_rot:
            r[f] = rot_all(getattr(sim,f),sim.g_dir)            
        self.grav_z = grav_z
        r['t'] = sim.t
        r['x'] = r['x'] - r['x'][0,:,:]
        
        for var in ['v','u','slip','x','vort','dudt','t']:
            r[var] = r[var][:-1]
            
        # make this dict an attribute; can be accessed via subscripting
        self.r = r
        
    def __getitem__(self,f):
        
        # map characteristic values to the variables they non-dimensionalize
        char_vals = {self.v_q:['v','u','slip'],
                     self.L_vf:['x'],
                     self.grav_z:['dudt','u_times_deldotu'],
                     self.T_vf:['t'],
                     self.grav_z:['press','drag','lift']}
        
        arr = self.r[f][self.ti_min:,...]
        if self.norm:
            for key in char_vals:
                if f in char_vals[key]:
                    return arr/key
        else:   
            return arr


def rot_coord_system(arr,g_dir):
    '''
    coordinate system rotation to align z with gravity, just for a single bubble.
    '''

    # z is in the direction of gravity
    z_dir = g_dir.copy()
    
    # x direction has to be normal to z, and we arbitrarily choose it's also normal to the DNS x
    x_dir_unscaled = np.cross(z_dir,[1,0,0])
    x_dir = x_dir_unscaled / np.linalg.norm(x_dir_unscaled)
    
    # get the y direction, has magnitude 1 since z and x are perpendicular each with magnitude 1
    y_dir = np.cross(z_dir,x_dir)
        
    arr_rot = np.array([np.dot(arr,x_dir),np.dot(arr,y_dir),np.dot(arr,z_dir)]).T

    return arr_rot

def rot_all(arrs,g_dirs,actually_rot=True):    
    '''
    rotate the coordinate systems for all n_b bubbles (along axis 1). 
    
    arrs has shape (n_t,n_b,3); g_dirs has shape (n_b,3)
    '''
    
    if actually_rot:
        arrs_new = []
        for i in np.arange(len(g_dirs)):
            arrs_new.append(rot_coord_system(arrs[:,i,:],g_dirs[i,:]))        
        arrs_new = np.moveaxis(np.array(arrs_new),0,1)
        return arrs_new
    else:
        return arrs

def get_vorticity(velgrad):
    # similar to the function in model.py
    velgrad_shape = np.shape(velgrad)
    vort_shape = velgrad_shape[:-1]
    vort = np.zeros(vort_shape)
    vort[...,0] = velgrad[...,2,1] - velgrad[...,1,2]
    vort[...,1] = velgrad[...,0,2] - velgrad[...,2,0]
    vort[...,2] = velgrad[...,1,0] - velgrad[...,0,1]
    return vort

def get_curvature(vel,t):
    accel = (np.gradient(vel,axis=0).T/np.gradient(t)).T
    curvature = np.linalg.norm(np.cross(vel,accel),axis=-1) / np.linalg.norm(vel,axis=-1)**3
    return curvature

'''
Helper functions for analysis
'''

def get_powerlaw(x,y,roll_window=1):
    powerlaw = np.gradient(np.log(y))/np.gradient(np.log(x))
    powerlaw = pd.Series(data=powerlaw).rolling(center=True,window=roll_window,min_periods=0).mean()
    return powerlaw

def get_minmax_series(df,varx,vary):

    x = df[varx].unique()
        
    low = np.zeros_like(x)
    high = np.zeros_like(x)
    
    for bi,b in enumerate(x):
        low[bi] = df[df[varx]==b][vary].min()
        high[bi] = df[df[varx]==b][vary].max()
        
    return x,low,high
    
def get_hist(y,bins=1001):
    '''return a normalized pdf and x locs of bin centers'''
    hist,edges = np.histogram(y[~np.isnan(y)],bins=bins,density=True)
    return edges[:-1]+np.diff(edges)/2, hist