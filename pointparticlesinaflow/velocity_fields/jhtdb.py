# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 16:50:22 2020

@author: danjr
"""

import numpy as np
from ..classes import VelocityField

try:
    from pyJHTDB import libJHTDB
    lJHTDB = libJHTDB()
    lJHTDB.initialize(exit_on_error=True)
    lJHTDB.add_token(r'edu.princeton.druth-29b2b7ef')
    lJHTDB_available = True
except ImportError:
    print('Unable to import pyJHTDB and initialize')
    lJHTDB_available = False
    
# isotropic 1024 coarse simulation parameters
ISOTROPIC1024COARSE_PARAMS = dict(
    u_rms = 0.686,
    L_int = 1.364,
    eta = 0.00280,
    T_eta = 0.0424,
    nu = 0.000185,
    lam = 0.113,    
    dx = 2*np.pi / 1024.,
    dt = 0.002, # the timestep at which the DNS data is stored, = 10*dt_orig
    dt_orig = 0.0002,
    t_min = 0.,
    t_max = 10.,
    )
ISOTROPIC1024COARSE_PARAMS['T_int'] = ISOTROPIC1024COARSE_PARAMS['L_int']/ISOTROPIC1024COARSE_PARAMS['u_rms']
ISOTROPIC1024COARSE_PARAMS['u_char'] = ISOTROPIC1024COARSE_PARAMS['u_rms']
ISOTROPIC1024COARSE_PARAMS['L_char'] = ISOTROPIC1024COARSE_PARAMS['L_int']
ISOTROPIC1024COARSE_PARAMS['T_char'] = ISOTROPIC1024COARSE_PARAMS['T_int']

# channel flow parameters
CHANNEL_PARAMS = dict(
    u_bulk = 1.,
    u_centerline = 1.1312,
    u_tau = 4.9968e-2,
    half_channel_height = 1,
    Re_bulk_fullheight = 3.9998e4,
    nu = 5e-5,
    pos_lims = ((-np.inf,-1,-np.inf),(np.inf,1,np.inf)),    
    t_min = 0.,
    t_max = 25.9935,
    dt = 0.0013,
    dt_orig = 0.0065,
    )
CHANNEL_PARAMS['u_char'] = CHANNEL_PARAMS['u_bulk']
CHANNEL_PARAMS['L_char'] = CHANNEL_PARAMS['half_channel_height']
CHANNEL_PARAMS['T_char'] = CHANNEL_PARAMS['L_char']/CHANNEL_PARAMS['u_char']

# put all dataset parameter dicts into one dict
JHTDB_DATASET_PARAMS = {'isotropic1024coarse':ISOTROPIC1024COARSE_PARAMS,
                        'channel':CHANNEL_PARAMS}

class JHTDBVelocityField(VelocityField):
    
    def __init__(self,data_set='isotropic1024coarse'):
        super().__init__(name='JHTDB_'+data_set)
        self.data_set = data_set
        if lJHTDB_available:
            self.lJHTDB = lJHTDB
        else:
            self.lJHTDB = None
        
        # store the parameters for this dataset
        if data_set in JHTDB_DATASET_PARAMS:
            [setattr(self,key,JHTDB_DATASET_PARAMS[data_set][key]) for key in JHTDB_DATASET_PARAMS[data_set]]
            
        new_save_vars = list(JHTDB_DATASET_PARAMS[data_set].keys())
        self._save_vars = self._save_vars + new_save_vars
            
    if lJHTDB_available:
        
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
        
    else:
        def get_velocity(self,t,x,):
            return None
        def get_velocity_gradient(self,t,x,lJHTDB=None):
            return None
        def get_dudt(self,t,x,u_t=None,delta_t=1e-4,lJHTDB=None):
            return None