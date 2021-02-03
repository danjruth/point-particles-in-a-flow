# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 11:26:45 2021

@author: druth
"""

import numpy as np
from .model import VelocityField

class RandomGaussianVelocityField(VelocityField):
    
    def __init__(self,n_modes=12,u_rms=1,L_int=1):
        VelocityField.__init__(self,name='random_gaussian')
        self.n_modes = n_modes
        self.u_rms = u_rms
        self.L_int = L_int        
        self.T_int = L_int/u_rms
    
        # random coefficients/wavenumbers/frequencies for each mode
        self.b = np.random.normal(scale=u_rms,size=(n_modes,3))
        self.c = np.random.normal(scale=u_rms,size=(n_modes,3))
        self.k = np.random.normal(scale=1./L_int,size=(n_modes,3))
        self.omega = np.random.normal(scale=1/self.T_int,size=(n_modes))
        
        def get_velocity(t,x):
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
        
        def get_velocity_gradient(t,x):
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
        
        def get_dudt(t,x,u_t=None):
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