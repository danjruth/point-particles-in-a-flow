# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 11:26:45 2021

@author: druth
"""

import numpy as np
from ..model import VelocityField
from .. import data

def load_random_gaussian_velocity_field(res):
    '''Return an instance of RandomGaussianVelocityField wth the parameters
    stored in/at res'''
    res = data.load_or_pass_on(res)
    vf = RandomGaussianVelocityField(n_modes=res['n_modes'],u_rms=res['u_rms'],L_int=res['L_int'])
    vf.load_field(res)
    return vf

class RandomGaussianVelocityField(VelocityField):
    
    def __init__(self,n_modes=12,u_rms=1,L_int=1):
        #VelocityField.__init__(self,name='random_gaussian')
        super().__init__(name='random_gaussian')
        self.n_modes = n_modes
        self.u_rms = u_rms
        self.L_int = L_int        
        self.T_int = L_int/u_rms
        
        self.u_char = u_rms
        self.L_char = L_int
        self.T_char = L_int/u_rms
        
        new_save_vars = ['b','c','k','omega','n_modes','u_rms','L_int','T_int']
        self._save_vars = self._save_vars + new_save_vars
        
    def init_field(self,):
        '''Generate random coefficients/wavenumbers/frequencies for each mode
        '''
        n_modes = self.n_modes
        u_rms = self.u_rms
        L_int = self.L_int
        T_int = self.T_int
    
        self.b = np.random.normal(scale=u_rms,size=(n_modes,3))
        self.c = np.random.normal(scale=u_rms,size=(n_modes,3))
        self.k = np.random.normal(scale=1./L_int,size=(n_modes,3))
        self.omega = np.random.normal(scale=1/T_int,size=(n_modes))
        
    def get_velocity(self,t,x):
        b = self.b
        c = self.c
        k = self.k
        omega = self.omega
        n_modes = self.n_modes
        
        vel = np.zeros((len(x),3))
        for m in range(n_modes):
            # outer product of the sin/cos term (len n_bubs) and the 3 coefficients for this mode gives shape (n_bubs,3)
            mode_contribution = np.outer(np.sin(np.dot(x,k[m,:])+omega[m]*t),b[m,:]) + np.outer(np.cos(np.dot(x,k[m,:])+omega[m]*t),c[m,:])
            vel = vel + mode_contribution
        return vel/np.sqrt(n_modes)
    
    def get_velocity_gradient(self,t,x):
        b = self.b
        c = self.c
        k = self.k
        omega = self.omega
        n_modes = self.n_modes
        
        velgrad = np.zeros((len(x),3,3))
        for m in range(n_modes):
            mode_contribution = np.zeros((len(x),3,3))
            for j in range(3):
                mode_contribution[:,:,j] = k[m,j]*np.outer(np.cos(np.dot(x,k[m,:])+omega[m]*t),b[m,:]) - k[m,j]*np.outer(np.sin(np.dot(x,k[m,:])+omega[m]*t),c[m,:])
            velgrad = velgrad + mode_contribution
        return velgrad/np.sqrt(n_modes)
    
    def get_dudt(self,t,x,u_t=None):
        b = self.b
        c = self.c
        k = self.k
        omega = self.omega
        n_modes = self.n_modes
        
        dudt = np.zeros((len(x),3))
        for m in range(n_modes):
            mode_contribution = omega[m]*np.outer(np.cos(np.dot(x,k[m,:])+omega[m]*t),b[m,:]) - omega[m]*np.outer(np.sin(np.dot(x,k[m,:])+omega[m]*t),c[m,:])
            dudt = dudt + mode_contribution
        return dudt/np.sqrt(n_modes)
    
def combine_Gaussian_fields(fields):
    '''Superpose many Gaussian fields
    '''
    combined_field = RandomGaussianVelocityField(n_modes=0,u_rms=np.nan,L_int=np.nan)
    combined_field.name = 'combined_field'    
    # concatenate each of the following variables from each field
    var_concat = ['b','c','k','omega']
    for var in var_concat:
        setattr(combined_field,var,np.concatenate([getattr(field,var) for field in fields],axis=0))
    combined_field.n_modes = len(combined_field.b)
    return combined_field