# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 14:59:40 2020

@author: danjr
"""

import numpy as np
import pyJHTDB
#from pyJHTDB.dbinfo import isotropic1024coarse
from pyJHTDB import libJHTDB
import pickle
import time as time_pkg

u_rms = 0.686
L_int = 1.364
eta = 0.00280

dt = 0.002
t_max = 10

lJHTDB = libJHTDB()
lJHTDB.initialize()

def get_velocity(t,x,lJHTDB=lJHTDB):
    return lJHTDB.getData(t, point_coords=x.astype(np.float32), data_set='isotropic1024coarse', getFunction='getVelocity', sinterp='Lag4', tinterp='PCHIPInt')

def myVelocityGradient(t,point_coords,delta=1e-4,lJHTDB=lJHTDB):
    '''
    Compute the velocity gradient numerically. Check the effect of delta, it should be around 1e-4.
    '''
    points_plus = np.zeros((len(point_coords),3,2))
    for i in np.arange(3):
        points_plus[:,i,0] = point_coords[:,i]-delta
        points_plus[:,i,1] = point_coords[:,i]+delta
    
    points_flat = np.array([[points_plus[:,0,0],point_coords[:,1],point_coords[:,2]],
                            [points_plus[:,0,1],point_coords[:,1],point_coords[:,2]],
                            [point_coords[:,0],points_plus[:,1,0],point_coords[:,2]],
                            [point_coords[:,0],points_plus[:,1,1],point_coords[:,2]],
                            [point_coords[:,0],point_coords[:,1],points_plus[:,2,0]],
                            [point_coords[:,0],point_coords[:,1],points_plus[:,2,1]]])
    
    points_flat = np.moveaxis(points_flat,-1,0)    
    points_flat = np.reshape(points_flat, (len(point_coords)*6,3))

    u_flat = get_velocity(t,point_coords,lJHTDB=lJHTDB)
    u = np.reshape(u_flat,(len(point_coords),6,3))
    
    vel_grad = np.zeros((len(point_coords),3,3)) # [point,component,grad_dir]    
    for j in np.arange(len(point_coords)):
        # for each point in time        
        for i in np.arange(3):
            # for each velocity direction
            vel_grad[j,i,0] = (u[j,1,i]-u[j,0,i])/(2*delta)
            vel_grad[j,i,1] = (u[j,3,i]-u[j,2,i])/(2*delta)
            vel_grad[j,i,2] = (u[j,5,i]-u[j,4,i])/(2*delta)
            
    return vel_grad

def get_vorticity(velgrad):
    vort = np.zeros((len(velgrad),3)) # 
    vort[:,0] = velgrad[...,2,1] - velgrad[...,1,2]
    vort[:,1] = velgrad[...,0,2] - velgrad[...,2,0]
    vort[:,2] = velgrad[...,1,0] - velgrad[...,0,1]
    return vort

def a_bubble(u,v,velgrad,dudt,d,Cd,Cm,Cl,g):
    '''
    calculate a bubble's accceleration given its velocity, the local water
    velocity, and the bubble size
    '''
    
    vort = get_vorticity(velgrad)
    slip = v - u
    
    # pressure force and added mass, in terms of the carrier velocity field
    u_times_deldotu = np.array([np.sum(velgrad[...,0,:]*u,axis=-1),
                                np.sum(velgrad[...,1,:]*u,axis=-1),
                                np.sum(velgrad[...,2,:]*u,axis=-1)]).T
    press = (1+Cm) * (d/2)**3*4./3*np.pi * (dudt + u_times_deldotu)
    
    # bouyant force
    grav = g*(d/2)**3*4./3*np.pi * np.array([0,0,1])
    
    # drag force    
    drag = -1 * Cd*0.5*np.pi*(d/2)**2 * (slip.T*np.linalg.norm(slip,axis=-1)).T
    
    # lift force
    lift = -1 * Cl * np.cross(slip,vort) * (d/2)**3*4./3*np.pi
    
    # calculate the added mass and the bubble acceleration
    m_added = Cm*(d/2)**3*4./3*np.pi
    a = (press+drag+grav+lift)/m_added
    
    return a

def quiescent_speed(d,g,Cd):
    return np.sqrt(8./3*(d/2) * g /Cd)

class PointBubbleSimulation:
    
    def __init__(self,params,fpath_save=None):
        
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
        self.t = np.arange(0,t_max,self.dt_use)
        self.n_t = len(self.t)
        
        if fpath_save is None:
            self.fpath_save = 'res_beta'+'{:05.4f}'.format(self.beta)+'_A'+'{:05.4f}'.format(self.A)+'_Cm'+'{:03.2f}'.format(self.Cm)+'_Cl'+'{:03.2f}'.format(self.Cl)+'_Cd'+'{:03.2f}'.format(self.Cd)+'_dtFactor'+'{:05.4f}'.format(self.dt_factor)+'.pkl'
        else:
            self.fpath_save = fpath_save
            
    def save(self,fpath_save=None):
        if fpath_save is None:
            fpath_save = self.fpath_save
        
        save_vars = ['beta','A',
                     'Cm','Cl','Cd',
                     'g','v_q','d',
                     'n_bubs','dt_factor','dt_use','t','n_t',
                     'x','v','u','dudt','velgrad']
        
        res = {attr:self.attr for attr in save_vars}
        with open(fpath_save, 'wb') as handle:
            pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        
    def init_sim(self):
        '''
        Initialize the simulation
        '''
        
        n_t = self.n_t
        n_bubs = self.n_bubs
        
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
        
    def _advance(self,ti):
        
        t = self.t[ti]
        u = self.u
        v = self.v
        x = self.x
        dudt = self.dudt
        velgrad = self.velgrad
        
        # liquid velocity
        u[ti+1,...] = get_velocity(t,x[ti,...],lJHTDB=lJHTDB)
        if ti==0:
            u[0,...] = u[1,...]
        
        # time velocity gradient
        delta = 1e-4
        u_deltat = get_velocity(t+delta,x[ti,...],lJHTDB=lJHTDB)
        dudt[ti+1,...] = (u_deltat-u[ti+1,...])/delta # future velocity at this point minus current velocity at this point
        
        # liquid velocity gradient
        velgrad[ti+1,...] = myVelocityGradient(t,x[ti,...])
        
        # start the simulation with v = u + v_q
        if ti==0:
            v[0,...] = u[1,...] + np.array([0,0,self.v_q])
        
        # bubble acceleration and new velocity
        a = a_bubble(u[ti+1,...],v[ti,...],velgrad[ti+1,...],dudt[ti+1,...])
        v[ti+1,...] = v[ti,...]+a*self.dt_use
        
        # new position
        x[ti+1,...] = x[ti,...]+v[ti+1,...]*self.dt_use
        
        # store the data
        self.u = u
        self.v = v
        self.dudt = dudt
        self.velgrad = velgrad
        
    def run_model(self,save_every=1000):
        
        while self.ti < self.n_t:
            
            # advance teh simulation
            t_start = time_pkg.time()
            self._advance(self.ti)
            print('Iteration '+str(self.ti)+', time '+'{:06.4f}'.format(self.t[self.ti])+', took '+'{:01.4f}'.format(time_pkg.time()-t_start)+' s.')
            self.ti = self.ti + 1
            
            # save, if necessary
            if (self.ti % save_every) == 0:        
                self.save()