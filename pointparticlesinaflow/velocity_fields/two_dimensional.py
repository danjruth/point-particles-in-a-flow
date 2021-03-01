# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 13:04:52 2021

@author: druth
"""

import numpy as np
from ..classes import VelocityField

class SteadyPlanarPoiseuilleFlow(VelocityField):
    '''2-D parabolic velocity profile, with x the flow direction, y the wall-
    normal direction, and z the into-plane direction. The centerline velocity 
    (at y=0) is u_center*e_x and the velocity at the walls (y=+-L/2) is 0. 
    '''
    
    def __init__(self,u_center=1,L=1,):
        super().__init__(name='planar_poiseuille')
        self.u_center=u_center
        self.L=L
        self.pos_lims = ((-np.inf,-L/2,-np.inf),(np.inf,L/2,np.inf))
        
        self.u_char = u_center
        self.T_char = L/u_center
        self.L_char = L
        
        new_save_vars = ['u_center','L']
        self._save_vars = self._save_vars + new_save_vars
        
    def _nan_outside_channel(self,x,arr):
        y = x[:,1]
        arr[y>self.L/2,...] = np.nan
        arr[y<-self.L/2,...] = np.nan
        return arr
        
    def get_velocity(self,t,x):        
        vel = np.zeros((len(x),3))
        y = x[:,1]
        vel[...,0] = self.u_center * ( 1 - (y/(self.L/2))**2 )
        # set velocity to nan outside the channel
        vel = self._nan_outside_channel(x, vel)        
        return vel
    
    def get_dudt(self,t,x,u_t=None):
        dudt = np.zeros_like(x)
        dudt = self._nan_outside_channel(x,dudt)
        return dudt
    
    def get_velocity_gradient(self,t,x):
        y = x[:,1]
        velgrad = np.zeros((len(x),3,3))
        velgrad[:,0,1] = -8 * y / self.L**2
        velgrad = self._nan_outside_channel(x, velgrad)
        return velgrad
    
class BlasiusBoundaryLayer(VelocityField):
    '''Blasius solution for the boundary layer over a flat plate with a steady
    flow. Flow is in the x direction, and the boundary layer starts at x=0. The
    wall-normal direction is y.
    '''
    pass