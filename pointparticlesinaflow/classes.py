# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 14:59:40 2020

@author: danjr
"""

import numpy as np
import pickle
from scipy.spatial.transform import Rotation
from pointparticlesinaflow import data, analysis
#from .equations import quiescent_speed

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
        self.pos_lims = ((-np.inf,-np.inf,-np.inf),(np.inf,np.inf,np.inf))
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
        self.velgrad = velocity_field.get_velocity_gradient(t,x)
        self.dudt = velocity_field.get_dudt(t,x,u_t=self.u)
        
'''
Class for the equation of motion
'''

class Force:
    
    def __init__(self,name='no_force',short_name='no_force'):
        self.name = name
        self.short_name = short_name
        self.pkeys = []
        self.precalcs = []
    
    def __call__(self,p):
        return np.nan
        
class EquationOfMotion:
    '''the __call__ method returns the new particle velocities, given
    parameters and data in the dict p
    '''
    
    additional_pkeys = []
    
    def __init__(self,name='no_forces',forces=[]):
        self.name = name
        self.forces = forces
        
        # get list of particle params involved in each force
        p = [f.pkeys for f in self.forces]        
        self.p = list({x for l in p for x in l})  + self.additional_pkeys
        
        # get list of precalcs
        precalcs = [f.precalcs for f in self.forces]
        self.precalcs = list({x for l in precalcs for x in l})
        
        self.force_names = [f.name for f in self.forces]
        self.force_short_names = [f.short_name for f in self.forces]
    
    def calc_m_eff(self,r):
        '''Calculate the effective particle mass by which to divide the sum of
        the forces in order to calculate the particle acceleration.
        '''
        return np.nan
        
    def _pre_calculations(self,r):
        '''update the dict r by performing some calculations on it (ie adding
        an entry "vort" which is the vorticity, based on the entry "velgrad")
        '''
        #return r
        for pc in self.precalcs:
            r = pc(r)
        return r
    
    def __call__(self,r,dt):
        r = self._pre_calculations(r)
        forces = [f(r) for f in self.forces]
        #print(np.shape(forces))
        sum_forces = np.sum(forces,axis=0)
        #print(np.shape(sum_forces))
        a = np.moveaxis(sum_forces,0,-1) / self.calc_m_eff(r)
        a = np.moveaxis(a,-1,0)
        #print(np.shape(a))
        return r['v']+a*dt
    
'''
Class for a simulation
'''

class Simulation:
    
    def __init__(self,velocity_field,equation_of_motion,particle_params,simulation_params,fpath=None):
        
        self.vf = velocity_field
        self.eom = equation_of_motion
        self.p = particle_params
        self.s = simulation_params
        
        if fpath is None:
            fpath = 'test_save.pkl'
        self.fpath=fpath
        
        # multiple integrations per timestep
        if 'n_call_per_timestep' not in self.s:
            self.s['n_call_per_timestep'] = 1
        self.s['dt_int'] = self.s['dt']/self.s['n_call_per_timestep']
        
        
    @property
    def pkeys(self):
        return list(self.p.keys())
        
    def init_sim(self):
        
        self.t = np.arange(self.s['t_min'],self.s['t_max'],self.s['dt'])
        self.s['n_t'] = len(self.t)
        self.ti = 0
        
        # positions and velocities of particles
        self.x = np.zeros((self.s['n_t'],self.s['n'],3)).astype(float)
        self.v = np.zeros_like(self.x)
        
        # velocity field
        self.u = np.zeros_like(self.x)
        self.dudt = np.zeros_like(self.x)
        self.velgrad = np.zeros((self.s['n_t'],self.s['n'],3,3))
        
    def _update(self,t,x,v):
        '''
        Calculate the new position and velocities given current time, position,
        and velocities
        '''
        
        fs = self.vf.get_field_state(t,x)
        
        # add entries to r
        r = {'v':v}
        for key in ['u','dudt','velgrad',]:
            r[key] = getattr(fs,key)
        # for each param listed in the necessary ones for the eom
        for key in self.eom.p:
            r[key] = self.p[key]
            
        v_new = self.eom(r,self.s['dt_int']) # based on everything at this point in time
        x_new = x+v_new*self.s['dt_int']
        
        return x_new, v_new, r

    def _advance(self,ti):
        
        x_old = self.x[ti,...]
        v_old = self.v[ti,...]
        for tii in range(self.s['n_call_per_timestep']):
            t_val = self.t[ti] + tii*self.s['dt']/self.s['n_call_per_timestep']
            x_old, v_old, r = self._update(t_val,x_old,v_old)
        x_new = x_old
        v_new = v_old 
        
        # for now, limit the x data to the values in eom.pos_lims
        for i in range(3):
            x_new[x_new[:,i]<self.vf.pos_lims[0][i],i] = self.vf.pos_lims[0][i]
            x_new[x_new[:,i]>self.vf.pos_lims[1][i],i] = self.vf.pos_lims[1][i]

        # store the data
        self.u[ti+1,...] = r['u'].copy() # assigning field state at t[ti] to ti+1?
        self.v[ti+1,...] = v_new.copy()
        self.x[ti+1,...] = x_new.copy()
        self.dudt[ti+1,...] = r['dudt'].copy()
        self.velgrad[ti+1,...] = r['velgrad'].copy()
        
    def run(self,disp=False,save_every=np.nan):
        
        for ti in np.arange(self.ti,self.s['n_t']-1,1):
            if disp:
                print('... time '+str(self.t[ti])+'/'+str(self.s['t_max']))
            self._advance(ti)
            self.ti = ti
            
            if ti%save_every == 0:
                self.save_dict()
            
    def to_dict(self):
        '''Put the bubble parameters, simulation parameters, and results of 
        the simulation in a dict. Save just the names of the velocity field
        and the equation of motion (since these classes can't be pickled
        reliably)
        '''        
        attrs_save = ['s','p',
                      't','ti',
                      'u','v','x','dudt','velgrad']
        d = {attr:getattr(self,attr) for attr in attrs_save}        
        d['PARAMS_vf'] = self.vf.to_dict()        
        return d
    
    def save_dict(self,fpath=None):
        if fpath is None:
            fpath = self.fpath
        d = self.to_dict()
        with open(fpath,'wb') as f:
            pickle.dump(d,f)
            
    def from_dict(self,d=None):
        if d is None:
            d = self.fpath
        if type(d) is str:
            d = pickle.load(open(d,'rb'))
        for key in d:
            setattr(self,key,d[key])
        for key in d['PARAMS_vf']:
            setattr(self.vf,key,d['PARAMS_vf'][key])

class SimulationOld:
    '''
    Main class for the simulation of point particles in flows.
    
    Parameters
    ----------
    
    velocity_field : pointparticlesinaflow.classes.VelocityField
        The velocity field to use for the simulations
        
    phys_params : dict
        The physical parameters of the simulation.
        
    sim_params : dict
        Parameters to use for the simulation
        
    eom : pointparticlesinaflow.classes.EquationOfMotion
        The equation of motion to use for the point particles.
    '''
    
    _save_vars = ['phys_params','sim_params','_save_vars',
                  'g_dir',
                  'x','v','u','dudt','velgrad','t','n_t',
                  'ti']
    
    def __init__(self,velocity_field,phys_params,sim_params,eom):
        
        self.velocity_field = velocity_field
        self.phys_params = phys_params
        self.sim_params = sim_params
        self.eom = eom
        
    def init_sim(self,g_dir='-z',pos_lims=((0,0,0),(2*np.pi,2*np.pi,2*np.pi)),vz_0=0):
        '''
        Initialize the simulation. Should only be called when the simulation is
        first created, not when it's being reloaded from some intermediate
        point. In that case, use add_data().
        
        Parameters
        ----------
        
        g_dir : str or np.ndarray
            The gravitational direction. If 'random', it is chosen randomly. If
            '-x', '-y', or '-z', it is antiparallel to either the x, y, or z
            axis. If np.ndarray, the direction of gravity for each bubble, with
            shape (n_bubs,3).
            
        pos_lims : tuple
            The spatial limits of the simulation, as
            ((min_x,min_y,min_z),(max_x,max_y,max_z)).
            
        vz_0 : float or str
            If float, the vertical component of the initial velocity. If 'v_q',
            the quiescent rise velocity is used (upwards, positive z); if
            '-v_q', the quiescent settling velocity (downwards, negative z) is
            used. The intial velocity can be changed to other values by 
            updating the .v attribute after init_sim() is called.      
        '''
        
        self = assign_attributes(self,self.phys_params,self.sim_params)
        
        # initial setup
        self.t = np.arange(self.t_min,self.t_max,self.dt)
        self.n_t = len(self.t)
        
        n_t = self.n_t
        n_bubs = self.n_bubs
        
        if vz_0 == 'v_q':
            vz_0 = self.v_q
        if vz_0 == '-v_q':
            vz_0 = self.v_q * -1
        
        # define the direction of gravity for each bubble
        gdir_dict = {'-x':0,'-y':1,'-z':2}
        if g_dir == 'random':
            self.g_dir = np.array([Rotation.random(1).apply([0,0,1]) for _ in range(n_bubs)])[:,0,:]        
        elif g_dir in gdir_dict:
            self.g_dir = np.zeros((n_bubs,3))
            self.g_dir[:,gdir_dict[g_dir]] = -1
        else:
            self.g_dir = g_dir
        
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
        
    def _construct_update_dict(self,ti):
        '''Construct the dict p, which will contain the data with which the 
        equation of motion is evaluated
        '''
        
        # get the field state
        fs = self.velocity_field.get_field_state(self.t[ti],self.x[ti,...])

        # add entries to p
        p = {'v':self.v[ti,...]}
        for key in ['u','dudt','velgrad',]:
            p[key] = getattr(fs,key)
        for key in self.eom.const_params:
            p[key] = getattr(self,key)
        
        return p
        
    def _advance(self,ti):
        '''
        Advance the simulation one timestep (to timestep ti) using forward
        Euler integration.
        '''
        
        p = self._construct_update_dict(ti)
        v_new = self.eom(p,self.dt) # based on everything at this point in time
        x_new = self.x[ti,...]+v_new*self.dt
        
        # for now, limit the x data to the values in eom.pos_lims
        for i in range(3):
            x_new[x_new[:,i]<self.velocity_field.pos_lims[0][i],i] = self.velocity_field.pos_lims[0][i]
            x_new[x_new[:,i]>self.velocity_field.pos_lims[1][i],i] = self.velocity_field.pos_lims[1][i]

        # store the data
        self.u[ti+1,...] = p['u'].copy() # assigning field state at t[ti] to ti+1?
        self.v[ti+1,...] = v_new.copy()
        self.x[ti+1,...] = x_new.copy()
        self.dudt[ti+1,...] = p['dudt'].copy()
        self.velgrad[ti+1,...] = p['velgrad'].copy()
        
    def run(self,save_every=100,fpath=None,disp=False):
        '''
        Run the simulation, starting from the current timestep stored in .ti.
        
        Parameters
        ----------
        
        save_every : int
            The number of timesteps between each time .save() is called.
            
        fpath : str
            The filepath at which to save the simulation.
            
        disp : bool
            Whether or not to print out the timestep every timestep.
        '''
        for ti in np.arange(self.ti,self.n_t-1,1):
            if disp:
                print('... time '+str(self.t[ti])+'/'+str(self.t_max))
            self._advance(ti)
            self.ti = ti
            
            if fpath is not None and ti%save_every==0:
                self.save(fpath)
            
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
        
class TestConvergence:
    '''Class to test the effect of dt by creating multiple Simulations, each 
    differing only in dt.
    '''
    
    def __init__(self,velocity_field,phys_params,sim_params,eom,dt_vals):
        
        self.velocity_field = velocity_field
        self.phys_params = phys_params
        self.sim_params = sim_params
        self.eom = eom
        self.dt_vals = dt_vals
        
        sims = []
        for i,dt in enumerate(dt_vals):
            sim_params_dt = sim_params.copy()
            sim_params['dt'] = dt
            sim = Simulation(velocity_field,phys_params,sim_params_dt,eom)
            sim.init_sim()
            if i>0:
                sim.x[0,...] = sims[0].x[0,...]
                sim.g_dir = sims[0].g_dir
                sim.v[0,...] = sims[0].v[0,...]
            sims.append(sim)
        self.sims = sims
        
    def run_all(self):
        [sim.run() for sim in self.sims]
        self.complete_sims = {dt:analysis.CompleteSim(sim) for dt,sim in zip(self.dt_vals,self.sims)}
        
