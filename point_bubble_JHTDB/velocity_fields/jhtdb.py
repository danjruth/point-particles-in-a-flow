# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 16:50:22 2020

@author: danjr
"""

import numpy as np
from ..model import VelocityField

try:
    from pyJHTDB import libJHTDB
    lJHTDB = libJHTDB()
    lJHTDB.initialize(exit_on_error=True)
    lJHTDB_available = True
except ImportError:
    print('Unable to import pyJHTDB and initialize')
    lJHTDB_available = False

# only define the velocity field class for this if the interface to the JHTDB
# is available
if lJHTDB_available:
    
    class JHTDBVelocityField(VelocityField):
        
        def __init__(self,data_set='isotropic1024coarse'):
            #VelocityField.__init__(self,name='JHTDB_'+data_set)
            super().__init__(name='JHTDB_'+data_set)
            self.data_set = data_set
            self.lJHTDB = lJHTDB
            
            # store physical and simulation propertie
            if data_set=='isotropic1024coarse':
                self.u_rms = 0.686
                self.L_int = 1.364
                self.T_int = self.L_int/self.u_rms
                self.eta = 0.00280
                self.T_eta = 0.0424
                self.lam = 0.113
                
                self.u_char = self.u_rms
                self.L_char = self.L_int
                self.T_char = self.T_int
                
                self.dx = 2*np.pi / 1024.
                self.dt = 0.002 # the timestep at which the DNS data is stored, = 10*dt_orig
                self.dt_orig = 0.0002
                self.t_min = 0.
                self.t_max = 10.
            
        def get_velocity(self,t,x,lJHTDB=None):
            if lJHTDB is None:
                lJHTDB = self.lJHTDB
            return lJHTDB.getData(t, point_coords=x.copy().astype(np.float32), data_set=self.data_set, getFunction='getVelocity', sinterp='Lag4', tinterp='PCHIPInt')

        def get_velocity_gradient(self,t,x,lJHTDB=None):
            
            if lJHTDB is None:
                lJHTDB = self.lJHTDB
        
            # query the data
            result_grad = lJHTDB.getData(t, x.copy().astype(np.float32), data_set=self.data_set,sinterp='FD4Lag4', tinterp='PCHIPInt',getFunction='getVelocityGradient')
            
            # put it into the correct shape
            result_grad_new = np.zeros((len(x),3,3)).astype(np.float32)
            for i in range(len(x)):
                this_velgrad = np.array([result_grad[i,0:3],result_grad[i,3:6],result_grad[i,6:]])
                result_grad_new[i,...] = this_velgrad
                
            return result_grad_new
        
        def get_dudt(self,t,x,u_t=None,delta_t=1e-4,lJHTDB=None):
            
            if lJHTDB is None:
                lJHTDB = self.lJHTDB
            
            if u_t is None:
                u_t = self.get_velocity(t,x,)
            u_tplusdeltat = self.get_velocity(t+delta_t,x,)
            dudt = (u_tplusdeltat - u_t) / delta_t
            return dudt