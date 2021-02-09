# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 14:59:40 2020

@author: danjr
"""

import numpy as np
import pickle
from scipy.spatial.transform import Rotation
from . import data, equations

'''
Classes for the velocity field
'''

class VelocityField:
    '''Velocity field, with methods to get the velocity, dudt, and velocity
    gradient given t and x. Defaults to a quiescent velocity field. Derived 
    classes (for Gaussian psuedoturbulence or HIT from the JHTDB, for example)
    in the velocity_fields submodule.
    '''
    
    def __init__(self,name='quiescent'):
        self.name = name
        self.u_char = np.nan
        self.T_char = np.nan
        self.L_char = np.nan
        pass
    
    def init_field(self):
        '''This can be redefined for derived classes which need to be
        initialized.
        '''
        pass
        
    '''
    Functions to get velocity, dudt, and velocity gradient, which should be
    overwritten in derived classes for specific velocity fields.
    '''
    
    def get_velocity(self,t,x):
        print('getting quiescent velocity')
        return np.zeros((len(x),3))
    
    def get_dudt(self,t,x,u_t=None):
        return np.zeros((len(x),3))
    
    def get_velocity_gradient(self,t,x,):
        return np.zeros((len(x),3,3))
    
    '''
    Functions common to any derived class of VelocityField
    '''
    
    _save_vars = ['name','u_char','T_char','L_char']
    def to_dict(self):
        '''Put the attributes named in self._save_vars into a dict
        '''
        res = {attr:getattr(self,attr) for attr in self._save_vars}
        return res
    
    def save(self,fpath,include_velfield_params=True):
        '''put the results into a dict and pickle it
        '''
        res = self.to_dict()
        data.save_obj(res,fpath)
        
    def load_field(self,res):
        '''
        Add velocity field data 
        '''
        res = data.load_or_pass_on(res)
        [setattr(self,key,res[key]) for key in self._save_vars]
    
    def get_field_state(self,t,x):
        '''get the state of the field at a point in time at given locations
        '''
        return FieldState(self,t,x)
    
    def get_2d_velocity_field(self,t,X,Y,Z):
        XYZ = np.array([X.flatten(),Y.flatten(),Z.flatten()]).T
        vel = self.get_velocity(t,XYZ)
        vel = np.reshape(vel,(np.shape(X)[0],np.shape(X)[1],3))
        return vel
    
    def get_2d_velocity_gradient_field(self,t,X,Y,Z):
        XYZ = np.array([X.flatten(),Y.flatten(),Z.flatten()]).T
        velgrad = self.get_velocity_gradient(t,XYZ)
        velgrad = np.reshape(velgrad,(np.shape(X)[0],np.shape(X)[1],3,3))
        return velgrad
    
    def calc_u_rms(self,n_t=10,t_lims=(0,1000),n_x=1000,x_lims=((0,0,0),(1000,1000,1000)),componentwise=False):
        '''Sample velocities at random times and locations to get u_rms.
        '''
        
        velocities = []
        for _ in range(n_t):
            x = np.random.uniform(low=x_lims[0],high=x_lims[1],size=(n_x,3))
            vels = self.get_velocity(np.random.uniform(t_lims[0],t_lims[1]),x)
            velocities.append(vels)
        velocities=np.array(velocities)
        if not componentwise:
            return velocities.std()
        else:
            return velocities.std(axis=(1,0))
        
    def calc_mean_flow(self,n_t=10,t_lims=(0,1000),n_x=1000,x_lims=((0,0,0),(1000,1000,1000))):
        velocities = []
        for _ in range(n_t):
            x = np.random.uniform(low=x_lims[0],high=x_lims[1],size=(n_x,3))
            vels = self.get_velocity(np.random.uniform(t_lims[0],t_lims[1]),x)
            velocities.append(vels)
        velocities=np.array(velocities)
        return np.mean(velocities,axis=(0,1))
    
    def calc_structure_function_x(self,delta_x_max=None,delta_x_min=1e-4,
                                  n_x=101,n_y=101,
                                  n_t=1000,t_lims=(0,1000),
                                  x_lims=((0,0,0),(1000,1000,1000))):
        '''Compute the structure function in the x direction
        '''
        if delta_x_max is None:
            delta_x_max = 100*self.L_char
        x = np.concatenate([[0],np.geomspace(delta_x_min,delta_x_max,n_x)])
        y = np.linspace(0,x_lims[1][1])
        X,Y = np.meshgrid(x,y)
        Z = np.zeros_like(X)
        mean_veldiffsqs = []
        for t in np.random.uniform(t_lims[0],t_lims[1],n_t):
            xyz_flat = np.array([X.flatten()+np.random.uniform(x_lims[0][0],x_lims[1][0]),Y.flatten()+np.random.uniform(x_lims[0][0],x_lims[1][0]),Z.flatten()+np.random.uniform(x_lims[0][2],x_lims[1][2])]).T
            vel_flat = self.get_velocity(t,xyz_flat)
            vel = np.reshape(vel_flat,(np.shape(X)[0],np.shape(X)[1],3))
            veldiffsq = (vel[:,:,0].T - vel[:,0,0]).T**2
            mean_veldiffsq = np.mean(veldiffsq,axis=0)
            mean_veldiffsqs.append(mean_veldiffsq)
        mean_veldiffsqs = np.mean(np.array(mean_veldiffsqs),axis=0)
        return x, mean_veldiffsqs
    
    def calc_autocorrelation_x(self,delta_x_max=None,delta_x_min=1e-4,
                                n_x=101,n_y=101,
                                n_t=1000,t_lims=(0,1000),
                                x_lims=((0,0,0),(1000,1000,1000))):
        if delta_x_max is None:
            delta_x_max = 100*self.L_char
        x = np.concatenate([[0],np.geomspace(delta_x_min,delta_x_max,n_x)])
        y = np.linspace(0,x_lims[1][1])
        X,Y = np.meshgrid(x,y)
        Z = np.zeros_like(X)
        mean_prod_vels = []
        for t in np.random.uniform(t_lims[0],t_lims[1],n_t):
            xyz_flat = np.array([X.flatten()+np.random.uniform(x_lims[0][0],x_lims[1][0]),Y.flatten()+np.random.uniform(x_lims[0][0],x_lims[1][0]),Z.flatten()+np.random.uniform(x_lims[0][2],x_lims[1][2])]).T
            vel_flat = self.get_velocity(t,xyz_flat)
            vel = np.reshape(vel_flat,(np.shape(X)[0],np.shape(X)[1],3))
            prod_vel = (vel[:,:,0].T * vel[:,0,0]).T
            mean_prod_vel = np.mean(prod_vel,axis=0)
            mean_prod_vels.append(mean_prod_vel)
        mean_prod_vels = np.mean(np.array(mean_prod_vels),axis=0)
        mean_prod_vels = mean_prod_vels / mean_prod_vels[0]
        return x, mean_prod_vels
        
    
    def get_velocity_gradient_numerical(self,t,x,delta_x=1e-4):
        '''Calculate the velocity gradient numerically
        '''
        velgrad = np.zeros((len(x),3,3,))
        # for the three directions in which to take the gradient
        for j in range(3):
            stencil = np.zeros((3))
            stencil[j] = 1
            vel_below = self.get_velocity(t,x-stencil*delta_x)
            vel_above = self.get_velocity(t,x+stencil*delta_x)
            velgrad[:,:,j] = (vel_above-vel_below)/(2*delta_x)
        return velgrad
    
    def get_dudt_numerical(self,t,x,delta_t=1e-4):
        '''Calculate the velocity gradient wrt time numerically
        '''
        vel_before = self.get_velocity(t-delta_t,x)
        vel_after = self.get_velocity(t+delta_t,x)
        return (vel_after-vel_before)/(2*delta_t)
        

class FieldState:
    '''velocity values at a given time and locations
    '''
    
    def __init__(self,velocity_field,t,x):
        
        self.t = t
        self.x = x
        self.u = velocity_field.get_velocity(t,x)
        self.dudt = velocity_field.get_dudt(t,x,u_t=self.u)
        self.velgrad = velocity_field.get_velocity_gradient(t,x)
        
'''
Class for the equation of motion
'''
        
class EquationOfMotion:
    '''the __call__ method returns the new particle velocities, given their
    old velocities, the field state at their locations, the bubble parameters,
    and teh timestep'''
    
    def __init__(self,name='no_forces'):
        self.name = name
    
    def __call__(self,v,fs,sim,dt):
        '''by default just return the old velocity'''
        return v.copy()
   
    
'''
Class for a simulation
'''

def assign_attributes(obj,phys_params,sim_params):
    '''
    Make the keys in phys_params and sim_params attributes of obj, and add
    on additional computed attirbutes.
    '''
    
    # extract physical parameters
    for key in phys_params:
        setattr(obj,key,phys_params[key])
    
    # set the inertial quiescent speed, if appropriate
    if all(x in list(phys_params.keys()) for x in ['d','g','Cd']):
        obj.v_q = equations.quiescent_speed(obj.d,obj.g,obj.Cd)
    
    # extract simulation parameters
    for key in sim_params:
        setattr(obj,key,sim_params[key])
        
    return obj

class Simulation:
    
    _save_vars = ['phys_params','sim_params','_save_vars',
                  'g_dir',
                  'x','v','u','dudt','velgrad','t','n_t',
                  'ti']
    
    def __init__(self,velocity_field,phys_params,sim_params,eom):
        
        self.velocity_field = velocity_field
        self.phys_params = phys_params
        self.sim_params = sim_params
        self.eom = eom
        
    def init_sim(self,g_dir='z',pos_lims=((0,0,0),(2*np.pi,2*np.pi,2*np.pi)),vz_0=0):
        '''
        Initialize the simulation. Should only be called when the simulation is
        first created, not when it's being reloaded from some intermediate
        point. In that case, use add_data().
        '''
        
        self = assign_attributes(self,self.phys_params,self.sim_params)
        
        # initial setup
        self.t = np.arange(self.t_min,self.t_max,self.dt)
        self.n_t = len(self.t)
        
        n_t = self.n_t
        n_bubs = self.n_bubs
        
        if vz_0 == 'v_q':
            vz_0 = self.v_q
        
        # define the direction of gravity for each bubble
        if g_dir == 'random':
            self.g_dir = np.array([Rotation.random(1).apply([0,0,1]) for _ in range(n_bubs)])[:,0,:]
        elif g_dir == 'z':
            self.g_dir = np.zeros((n_bubs,3))
            self.g_dir[:,-1] = 1
        
        self.x = np.zeros((n_t,n_bubs,3))
        self.u = np.zeros((n_t,n_bubs,3))
        self.v = np.zeros((n_t,n_bubs,3))
        self.v[0,:,:] = vz_0 * self.g_dir
        self.velgrad = np.zeros((n_t,n_bubs,3,3))
        self.dudt = np.zeros((n_t,n_bubs,3))
        
        self.x[0,...] = np.random.uniform(low=pos_lims[0],high=pos_lims[1],size=(n_bubs,3))
                
        self.ti = 0
        self.t = np.arange(self.t_min,self.t_max,self.dt)
        self.n_t = len(self.t)
        
    
        
    def _advance(self,ti):
        
        # get the field state
        fs = self.velocity_field.get_field_state(self.t[ti],self.x[ti,...])

        # next velocity and position, given current field state and velocity
        v_new = self.eom(self.v[ti,...],fs,self,self.dt) # based on everything at this point in time
        x_new = self.x[ti,...]+v_new*self.dt

        # store the data
        self.u[ti+1,...] = fs.u.copy() # assigning field state at t[ti] to ti+1?
        self.v[ti+1,...] = v_new.copy()
        self.x[ti+1,...] = x_new.copy()
        self.dudt[ti+1,...] = fs.dudt
        self.velgrad[ti+1,...] = fs.velgrad
        
    def run(self):
        for ti in np.arange(self.ti,self.n_t-1,1):
            self._advance(ti)
            self.ti = ti
            
    def add_data(self,res,include_velfield=False):
        '''
        Add partially-complete simulation data stored in the dict res
        '''
        res = data.load_or_pass_on(res)
        [setattr(self,key,res[key]) for key in self._save_vars]
        self.ti = max(self.ti-1,0)
        self = assign_attributes(self,self.phys_params,self.sim_params)
        if include_velfield:
            self.velocity_field.load_field(res['velfield_params'])
        
    def to_dict(self):
        '''Put the bubble parameters, simulation parameters, and results of 
        the simulation in a dict. Save just the names of the velocity field
        and the equation of motion (since these classes can't be pickled
        reliably)
        '''
        res = {attr:getattr(self,attr) for attr in self._save_vars}
        res['velocity_field_name'] = self.velocity_field.name
        res['equation_of_motion_name'] = self.eom.name
        return res
    
    def save(self,fpath,include_velfield_params=True):
        '''put the results into a dict and pickle it'''
        res = self.to_dict()
        res['velfield_params'] = self.velocity_field.to_dict()
        data.save_obj(res,fpath)