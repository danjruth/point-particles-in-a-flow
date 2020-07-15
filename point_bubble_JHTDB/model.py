# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 14:59:40 2020

@author: danjr
"""

import numpy as np
import pickle
import time as time_pkg
from point_bubble_JHTDB import interface
import os.path
from scipy.spatial.transform import Rotation


u_rms = 0.686
L_int = 1.364
eta = 0.00280

dt = 0.002 # the timestep at which the DNS data is stored, = 10*dt_orig
t_max_turbulence = 10

data_dir = r'/home/idies/workspace/Storage/danjruth/persistent/point_bubble_data//'

def get_vorticity(velgrad):
    vort = np.zeros((len(velgrad),3)) # 
    vort[:,0] = velgrad[...,2,1] - velgrad[...,1,2]
    vort[:,1] = velgrad[...,0,2] - velgrad[...,2,0]
    vort[:,2] = velgrad[...,1,0] - velgrad[...,0,1]
    return vort

def calc_pressure_force(u,velgrad,dudt,d,Cm):
    # pressure force and added mass, in terms of the carrier velocity field
    u_times_deldotu = np.array([np.sum(velgrad[...,0,:]*u,axis=-1),
                                np.sum(velgrad[...,1,:]*u,axis=-1),
                                np.sum(velgrad[...,2,:]*u,axis=-1)]).T
    press = (1+Cm) * (d/2)**3*4./3*np.pi * (dudt + u_times_deldotu)
    return press

def calc_grav_force(g,d,g_dir):
    return g*(d/2)**3*4./3*np.pi * g_dir

def calc_drag_force(slip,d,Cd):
    drag = -1 * Cd*0.5*np.pi*(d/2)**2 * (slip.T*np.linalg.norm(slip,axis=-1)).T
    return drag

def calc_lift_force(slip,vort,d,Cl):
    lift = -1 * Cl * np.cross(slip,vort) * (d/2)**3*4./3*np.pi
    return lift

def a_bubble(u,v,velgrad,dudt,d,Cd,Cm,Cl,g,g_dir):
    '''
    calculate a bubble's accceleration given its velocity, the local water
    velocity, and the bubble size
    '''
    
    vort = get_vorticity(velgrad)
    slip = v - u
    
    # pressure force
    press = calc_pressure_force(u,velgrad,dudt,d,Cm)
    
    # bouyant force
    grav = calc_grav_force(g,d,g_dir)
    
    # drag force    
    drag = calc_drag_force(slip,d,Cd)
    
    # lift force
    lift = calc_lift_force(slip,vort,d,Cl)
    
    # calculate the added mass and the bubble acceleration
    m_added = Cm*(d/2)**3*4./3*np.pi
    a = (press+drag+grav+lift)/m_added
    
    return a

def quiescent_speed(d,g,Cd):
    return np.sqrt(8./3*(d/2) * g /Cd)

class PointBubbleSimulation:
    
    def __init__(self,params,fname_save=None):
        
        self.params = params
        
        # bubble parameters
        self.beta = params['beta']
        self.A = params['A']
        self.Cm = params['Cm']
        self.Cl = params['Cl']
        self.Cd = params['Cd']
        self.g = (u_rms**2/L_int)/self.A
        self.v_q = u_rms/self.beta
        self.d = self.Cd*self.v_q**2*3./4 / self.g
                
        # simulation parameters
        self.n_bubs = params['n_bubs']
        self.dt_factor = params['dt_factor']
        self.dt_use = self.dt_factor*dt
        
        if 't_min' in list(params.keys()):
            self.t_min = params['t_min']
        else:
            self.t_min = 0
        if 't_max' in list(params.keys()):
            self.t_max = params['t_max']
        else:
            self.t_max = t_max_turbulence
            
        self.t = np.arange(self.t_min,self.t_max,self.dt_use)
        self.n_t = len(self.t)
        
        if fname_save is None:
            self.fname_save = 'res_beta'+'{:05.4f}'.format(self.beta)+'_A'+'{:05.4f}'.format(self.A)+'_Cm'+'{:03.2f}'.format(self.Cm)+'_Cl'+'{:03.2f}'.format(self.Cl)+'_Cd'+'{:03.2f}'.format(self.Cd)+'_randomOrientation.pkl'
        else:
            self.fname_save = fname_save
            
    def save(self,fname_save=None):
        if fname_save is None:
            fname_save = self.fname_save
        
        save_vars = ['beta','A',
                     'Cm','Cl','Cd',
                     'g','v_q','d','g_dir',
                     'n_bubs','dt_factor','dt_use','t','n_t',
                     'x','v','u','dudt','velgrad','ti']
        
        res = {attr:getattr(self,attr) for attr in save_vars}
        print('Saving data to '+fname_save)
        with open(data_dir+fname_save, 'wb') as handle:
            pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        
    def init_sim(self):
        '''
        Initialize the simulation
        '''
        
        n_t = self.n_t
        n_bubs = self.n_bubs
        
        # gravity direction chosen randomly for each bubble
        self.g_dir = np.array([Rotation.random(1).apply([0,0,1]) for _ in range(n_bubs)])[:,0,:] # gravity direction
        
        self.x = np.zeros((n_t,n_bubs,3))
        self.u = np.zeros((n_t,n_bubs,3))
        self.v = np.zeros((n_t,n_bubs,3))
        self.velgrad = np.zeros((n_t,n_bubs,3,3))
        self.dudt = np.zeros((n_t,n_bubs,3))
        
        self.x[0,...] = np.random.uniform(low=0,high=2*np.pi,size=(n_bubs,3))
                
        self.ti = 0
        
    def add_data(self,res):
        '''
        Add partially-complete simulation data stored in the dict res
        '''
        
        self.x = res['x']
        self.u = res['u']
        self.v = res['v']
        self.velgrad = res['velgrad']
        self.dudt = res['dudt']
        self.ti = res['ti']-1
        
    def add_data_if_existing(self,fname_save=None):
        if fname_save is None:
            fname_save = self.fname_save
            
        if os.path.isfile(data_dir+fname_save):
            print('Loading data from '+data_dir+fname_save)
            with open(data_dir+fname_save, 'rb') as handle:
                res = pickle.load(handle)
            self.add_data(res)
        else:
            print('Did not find file '+str(data_dir+fname_save))
        
    def _advance(self,ti):
        
        t = self.t[ti]
        u = self.u
        v = self.v
        x = self.x
        dudt = self.dudt
        velgrad = self.velgrad
        
        # liquid velocity
        u[ti+1,...] = interface.get_velocity(t,x[ti,...])
        if ti==0:
            u[0,...] = u[1,...]
        
        # time velocity gradient
        delta = 1e-4
        u_deltat = interface.get_velocity(t+delta,x[ti,...])
        dudt[ti+1,...] = (u_deltat-u[ti+1,...])/delta # future velocity at this point minus current velocity at this point
        
        # liquid velocity gradient
        velgrad[ti+1,...] = interface.get_velocity_gradient(t,x[ti,...])
        
        # start the simulation with v = u + v_q
        if ti==0:
            v[0,...] = u[1,...] + np.array([0,0,self.v_q])
        
        # bubble acceleration and new velocity
        a = a_bubble(u[ti+1,...],v[ti,...],velgrad[ti+1,...],dudt[ti+1,...],self.d,self.Cd,self.Cm,self.Cl,self.g,self.g_dir)
        v[ti+1,...] = v[ti,...]+a*self.dt_use
        
        # new position
        x[ti+1,...] = x[ti,...]+v[ti+1,...]*self.dt_use
        
        # store the data
        self.u = u
        self.v = v
        self.dudt = dudt
        self.velgrad = velgrad
        
    def run_model(self,save_every=500,n_try_max=10):
        
        print('running the model')
        
        while self.ti < self.n_t-1:
            
            n_tried = 0
            #while n_tried < n_try_max:
            if True:
            
                #try:
                if True:
                    # advance the simulation
                    t_start = time_pkg.time()
                    self._advance(self.ti)
                    print(self.fname_save+' : Timestep '+str(self.ti)+', time '+'{:06.4f}'.format(self.t[self.ti])+', took '+'{:01.4f}'.format(time_pkg.time()-t_start)+' s.')
                    self.ti = self.ti + 1
                    
                else:
                #except:
                    n_tried = n_tried+1
                    print('Pausing execution for 5 s...')
                    time_pkg.sleep(5)
            
            # save, if necessary
            if (self.ti % save_every) == 0:                
                self.save()
                
        self.save()
                
def load_sim_from_file(fpath):
    with open(fpath, 'rb') as handle:
        res = pickle.load(handle)
    p = PointBubbleSimulation(res)
    p.add_data(res)
    return(p)

default_params = {'beta':0.5,
                 'A':0.1,
                 'Cm':0.5,
                 'Cd':0.5,
                 'Cl':0.5,
                 'n_bubs':500,
                 'dt_factor':0.5,}
def run_model_default_params(changed_params):
    '''
    Specify and run a model which differs from the default parameters by changed_params
    '''
    
    model_is_done = False
    
    while not model_is_done:
        
        try:            
            params = default_params.copy()
            for key in list(changed_params.keys()):
                params[key] = changed_params[key]
            m = PointBubbleSimulation(params)
            m.init_sim()
            m.add_data_if_existing()
            m.run_model()
            model_is_done = True
            
        except:
            print('Program failed, starting again')

    