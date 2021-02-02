# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 10:59:12 2021

@author: druth
"""

import numpy as np

u_rms = 0.686
L_int = 1.364
T_int = L_int/u_rms
eta = 0.00280
T_eta = 0.0424
lam = 0.113
lam_by_Lint = lam / L_int

dx = 2*np.pi / 1024.
dt = 0.002 # the timestep at which the DNS data is stored, = 10*dt_orig
dt_orig = 0.0002
t_max_turbulence = 10

def A_given_dByL(d_by_L,beta,Cd):
    '''
    Calculate A when you want to specify d/L instead of A
    '''
    
    # calculate "physical" parameters
    d = d_by_L * L_int
    v_q = u_rms/beta
    g = (3./4) * Cd * v_q**2/d
    
    # calculate A
    A = u_rms**2 / (g*L_int)
    
    return A

default_params = {'beta':0.5,
                 'A':0.1,
                 'Cm':0.5,
                 'Cd':0.5,
                 'Cl':0.5,
                 'scale_by_d':False, # whether the pressure, lift terms should be scaled according to the d value chosen
                 'n_bubs':500,
                 'dt_factor':0.5,}
def run_model_default_params(changed_params,fname_save=None):
    '''
    Specify and run a model which differs from the default parameters by changed_params
    
    example command to run from a terminal (to run for d=lambda):
    python3 -c "from point_bubble_JHTDB.model import *; beta=0.5; Cd=0.5; Cl=0.25; A=A_given_dByL(lam_by_Lint,beta,Cd); run_model_default_params({'Cl':Cl,'beta':beta,'A':A,'Cd':Cd})"
    python3 -c "from point_bubble_JHTDB.model import *; beta=0.25; Cd=1; Cl=0.25; A=A_given_dByL(0.12,beta,Cd); run_model_default_params({'Cl':Cl,'beta':beta,'A':A,'Cd':Cd,'n_bubs':500})"
    python3 -c "from point_bubble_JHTDB.model import *; beta=0.25; Cd=1; Cl=0.0; A=A_given_dByL(0.2,beta,Cd); run_model_default_params({'Cl':Cl,'beta':beta,'A':A,'Cd':Cd,'n_bubs':500,'dt_factor':0.5})"
    '''

    params = default_params.copy()
    for key in list(changed_params.keys()):
        params[key] = changed_params[key]
    m = PointBubbleSimulation(params,fname_save=fname_save)
    m.init_sim()
    m.add_data_if_existing()
    m.run_model()

def run_for_matching_A(beta,d_by_L,A,Cl):
    
    '''
    python3 -c "from point_bubble_JHTDB.model import *; run_for_matching_A(beta,d_by_L,A,Cl)"
    '''
    
    d = d_by_L * L_int
    
    g = u_rms**2 / (L_int*A)
    
    v_q = u_rms / beta
    Cd = 4./3 * d * g / v_q**2
    
    run_model_default_params({'beta':beta,'A':A,'Cd':Cd,'Cl':Cl},fname_save=None)