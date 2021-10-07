# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 11:26:45 2021

@author: druth
"""

import numpy as np
from ..classes import VelocityField
from .. import data


class LinearDeepWave(VelocityField):
    
    def __init__(self,a=1,k=3,g=9.81):
        #VelocityField.__init__(self,name='random_gaussian')
        super().__init__(name='lineardeepwave')
        self.a = a
        self.k = k
        self.g = g
        self.omega = np.sqrt(self.k*self.g)
        self.wavelength = 2*np.pi/self.k
        
        self.u_char = self.omega/self.k
        self.L_char = 1./self.k
        self.T_char = 1./self.omega
        
        new_save_vars = ['a','k','g','omega']
        self._save_vars = self._save_vars + new_save_vars
        
        self.pos_lims = [[-np.inf,-np.inf,-np.inf],[np.inf,np.inf,a]]
        
    def init_field(self,):
        '''Generate random coefficients/wavenumbers/frequencies for each mode
        '''
        
    def wave_profile(self,x,t):
        '''compute the wave profile--here, x is just the x component of the 
        position
        '''
        return self.a * np.cos(self.k*x - self.omega*t)

    def get_velocity(self,t,x):
        a = self.a
        omega = self.omega
        k = self.k
        omega = self.omega        
        pref = omega * a * np.exp(k*x[...,2])
        
        u = pref * np.cos(k*x[...,0] - omega*t)
        v = np.zeros_like(u)
        w = pref * np.sin(k*x[...,0] - omega*t)
        vel = np.array([u,v,w]).T
        
        surface_profile = self.wave_profile(x[...,0],t)        
        vel[x[...,2]>=surface_profile] = np.nan
        
        return vel
    
    def get_velocity_gradient(self,t,x):
        a = self.a
        omega = self.omega
        k = self.k
        omega = self.omega        
        pref = omega * a * np.exp(k*x[...,2])
        
        dudx = pref * np.sin(k*x[...,0]-omega*t) * k * -1
        dudy = np.zeros_like(dudx)
        dudz = pref * k * np.cos(k*x[...,0]-omega*t)
        dudx = np.array([dudx,dudy,dudz])
        
        dvdx = np.zeros_like(dudx)
        
        dwdx = pref * np.cos(k*x[...,0]-omega*t) * k
        dwdy = np.zeros_like(dwdx)
        dwdz = pref * k * np.sin(k*x[...,0]-omega*t)
        dwdx = np.array([dwdx,dwdy,dwdz])
        
        arr = np.array([dudx,dvdx,dwdx])
        arr = np.moveaxis(arr,-1,0)
        
        return arr
    
    def get_dudt(self,t,x,u_t=None):
        a = self.a
        omega = self.omega
        k = self.k
        omega = self.omega        
        pref = omega * a * np.exp(k*x[...,2])
        dudt = pref * omega * np.sin(k*x[...,0]-omega*t)
        dvdt = np.zeros_like(dudt)
        dwdt = pref * omega * np.cos(k*x[...,0]-omega*t) * -1
        
        return np.array([dudt,dvdt,dwdt]).T
    
if __name__=='__main__':
    
    vf = LinearDeepWave(a=0.5,k=8)