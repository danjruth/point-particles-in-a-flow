# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 11:55:39 2020

@author: ldeike
"""

import numpy as np
import pickle
from . import classes
import pandas as pd

class CompleteSim():
    
    def __init__(self,sim,norm=False,rotated=True):
        
        # should arrays be normalized when returned
        self.norm = norm
        
        # are the vectors to be rotated s.t. z is parallel to gravity
        self.rotated = rotated
        
        # copy parameters from the Simulation
        self.velocity_field = sim.velocity_field
        self.phys_params = sim.phys_params
        self.sim_params = sim.sim_params
        self.eom = sim.eom
        self = classes.assign_attributes(self,self.phys_params,self.sim_params)
        
        # characteristic scales of the velocity field
        self.u_vf = sim.velocity_field.u_char
        self.L_vf = sim.velocity_field.L_char
        self.T_vf = sim.velocity_field.T_char
            
        # nondimensional numbers
        #self.dstar = self.d / self.L_vf
        #self.dstar_by_Cd = self.dstar / self.Cd
        #self.beta = self.u_vf / self.v_q
        #self.Fr = self.u_vf / np.sqrt(self.d*self.g)
        
        # get the forces and rotate everything so z is aligned with gravity for each bubble
        self._calc_forces(sim)
        self._rotate_and_store(sim,actually_rotate=rotated)
        self._remove_first_index(sim)
        
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
        
    def _calc_forces(self,sim):
        '''Compute the forces acting on the particle, and rotate all vectors to
        the gravity-aligned coordinate system
        '''
        
        # gravity, for normalizing forces
        if 'g' in sim.phys_params:
            vol = (sim.d/2.)**3 * 4./3 * np.pi
            self.grav_z = sim.g * vol
        
        # calculate forces over time
        p = {'u':sim.u,'v':sim.v,'dudt':sim.dudt,'velgrad':sim.velgrad}
        for key in sim.eom.const_params:
            p[key] = getattr(sim,key)
        p = sim.eom._pre_calculations(p)
        forces = {f.short_name:f(p) for f in sim.eom.forces}
        # add the time dimension to each force if it's not present
        for f in forces:
            if forces[f].ndim==2:
                forces[f] = np.array([forces[f]]*len(sim.t))
        # set force as attributes of sim
        [setattr(sim,key,forces[key]) for key in forces]
        
    def _rotate_and_store(self,sim,actually_rotate=True):
        
        forces = [f.short_name for f in sim.eom.forces]
                
        # store rotated numerical results in a dict
        r = {}
        fields_rot = ['v','u','x','dudt',]+forces # todo: include velgrad
        for f in fields_rot:
            r[f] = rot_all(getattr(sim,f),sim.g_dir,actually_rot=actually_rotate)            
        r['t'] = sim.t
        self.r = r
        
    def _remove_first_index(self,sim):
        
        forces = [f.short_name for f in sim.eom.forces]
        
        # get rid of the first index
        for var in ['v','u','x','dudt','t']+forces:
            self.r[var] = self.r[var][1:]
        
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
            arrs_new.append(rot_coord_system(arrs[:,i,:],g_dirs[i,:]*-1))        
        arrs_new = np.moveaxis(np.array(arrs_new),0,1)
        return arrs_new
    else:
        return arrs

def get_vorticity(velgrad):
    # similar to the function in equations.py
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
Functions for analysis/setting up simulations
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
    
def get_hist(y,bins=1001,cumulative=False):
    '''return a normalized pdf and x locs of bin centers'''
    hist,edges = np.histogram(y[~np.isnan(y)],bins=bins,density=True)
    if cumulative:
        hist = np.cumsum(hist*np.diff(edges))
    return edges[:-1]+np.diff(edges)/2, hist

def quiescent_speed(d,g,Cd):
    '''quiescent speed with constant Cd
    '''
    return np.sqrt(4./3 * d * g /Cd)

def quiescent_speed_visc(d,g,nu):
    '''using Cd=24/Re
    '''
    return 1./18 * d**2 * g / nu

def dg_given_nondim(Fr,dstar,u_vf,L_vf):
    '''calculate d and g given Fr and dstar
    '''
    d = dstar * L_vf
    g = u_vf**2 / (Fr**2*d)
    return d,g

def nu_given_Req(d,g,Cd_q,Re_q):
    '''calculate viscosity given the quiescent parameters
    '''
    v_q = quiescent_speed(d,g,Cd_q)
    nu = d*v_q / Re_q
    return nu

def nu_given_quiescent_visc(d,g,v_q):
    '''calulate the nu which yields v_q given d and g, assuming C_D = 24/Re
    '''
    return 1./18 * d**2 * g / v_q