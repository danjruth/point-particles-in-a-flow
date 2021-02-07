# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 14:59:40 2020

@author: danjr
"""

import numpy as np
import pickle
from scipy.spatial.transform import Rotation
from . import data

#data_dir = r'/home/idies/workspace/Storage/danjruth/persistent/point_bubble_data//'

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
        pass
    
    '''
    Functions to get velocity, dudt, and velocity gradient, which should be
    overwritten in derived classes for specific velocity fields.
    '''
    
    def get_velocity(self,t,x):
        print('getting quiescent velocity')
        return np.zeros((len(x),3))
    
    def get_dudt(self,t,x,u_t=None):
        return np.zeros((len(x),3))
    
    def get_velocity_gradient(self,t,x,):
        return np.zeros((len(x),3,3))
    
    '''
    Functions common to any derived class of VelocityField
    '''
    
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
        self.dudt = velocity_field.get_dudt(t,x,u_t=self.u)
        self.velgrad = velocity_field.get_velocity_gradient(t,x)
        
'''
Classes for the equations of motion
'''
        
class EquationOfMotion:
    '''the __call__ method returns the new particle velocities, given their
    old velocities, the field state at their locations, the bubble parameters,
    and teh timestep'''
    
    def __init__(self,name='no_forces'):
        self.name = name
    
    def __call__(self,v,fs,sim,dt):
        '''by default just return the old velocity'''
        return v.copy()
    
class MaxeyRileyPointBubbleConstantCoefs(EquationOfMotion):
    def __init__(self):
        EquationOfMotion.__init__(self,name='MaxeyRiley_pointbubble_constantcoefficients')
        
    def __call__(self,v,fs,sim,dt):
        '''calculate a new v based on the current v, the field state, and the
        bubble parameters stored in sim'''
        # (u,v,velgrad,dudt,d,Cd,Cm,Cl,g,g_dir)
        a = a_bubble_MR_constantcoefficients(fs.u,v,fs.velgrad,fs.dudt,sim.d,sim.Cd,sim.Cm,sim.Cl,sim.g,sim.g_dir)
        return v+a*dt
    
class MaxeyRileyPointBubbleVariableCoefs(EquationOfMotion):
    def __init__(self):
        EquationOfMotion.__init__(self,name='MaxeyRiley_pointbubble_variablecoefficients')
                
    def __call__(self,v,fs,sim,dt):
        '''calculate a new v based on the current v, the field state, and the
        bubble parameters stored in sim'''
        # (u,v,velgrad,dudt,d,Cd,Cm,Cl,g,g_dir)
        
        Re = np.linalg.norm(v-fs.u,axis=-1) * sim.d / sim.nu
        Cd = calc_Cd_Snyder(Re)        
        Cd_arr = np.ones((len(Cd),3))
        Cd_arr = (Cd_arr.T*Cd).T
        Cm = 0.5
        Cl = 0
        a = a_bubble_MR_constantcoefficients(fs.u,v,fs.velgrad,fs.dudt,sim.d,Cd_arr,Cm,Cl,sim.g,sim.g_dir)
        return v+a*dt
    
class LagrangianEOM(EquationOfMotion):
    def __init__(self):
        EquationOfMotion.__init__(self,name='Lagrangian')
        
    def __call__(self,v,fs,sim,dt):
        '''return the fluid velocity at the particle locations
        '''
        return fs.u
    
'''
Class for a simulation
'''

def assign_attributes(obj,bubble_params,sim_params):
    '''
    Make the keys in bubble_params and sim_params attributes of obj, and add
    on additional computed attirbutes.
    '''
    
    # extract bubble parameters
    for key in bubble_params:
        setattr(obj,key,bubble_params[key])
    obj.v_q = quiescent_speed(obj.d,obj.g,obj.Cd)
    
    # extract simulation parameters
    for key in sim_params:
        setattr(obj,key,sim_params[key])
        
    return obj

class Simulation:
    
    def __init__(self,velocity_field,bubble_params,sim_params,eom):
        
        self.velocity_field = velocity_field
        self.bubble_params = bubble_params
        self.sim_params = sim_params
        self.eom = eom
        
        self = assign_attributes(self,bubble_params,sim_params)
        
        # initial setup
        self.t = np.arange(self.t_min,self.t_max,self.dt)
        self.n_t = len(self.t)
        
    def init_sim(self,g_dir='random',pos_lims=((0,0,0),(2*np.pi,2*np.pi,2*np.pi)),vz_0=0):
        '''
        Initialize the simulation. Should only be called when the simulation is
        first created, not when it's being reloaded from some intermediate
        point.
        '''
        
        n_t = self.n_t
        n_bubs = self.n_bubs
        
        if vz_0 == 'v_q':
            vz_0 = self.v_q
        
        # define the direction of gravity for each bubble
        if g_dir == 'random':
            self.g_dir = np.array([Rotation.random(1).apply([0,0,1]) for _ in range(n_bubs)])[:,0,:]
        elif g_dir == 'z':
            self.g_dir = np.zeros((n_bubs,3))
            self.g_dir[:,-1] = 1
        
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
        self.g_dir = res['g_dir']
        
    def _advance(self,ti):
        
        # get the field state
        fs = self.velocity_field.get_field_state(self.t[ti],self.x[ti,...])

        # next velocity and position, given current field state and velocity
        v_new = self.eom(self.v[ti,...],fs,self,self.dt) # based on everything at this point in time
        x_new = self.x[ti,...]+v_new*self.dt

        # store the data
        self.u[ti+1,...] = fs.u.copy() # assigning field state at t[ti] to ti+1?
        self.v[ti+1,...] = v_new.copy()
        self.x[ti+1,...] = x_new.copy()
        self.dudt[ti+1,...] = fs.dudt
        self.velgrad[ti+1,...] = fs.velgrad
        
    def run(self):
        for ti in np.arange(self.ti,self.n_t-1,1):
            self._advance(ti)
            self.ti = ti
        
    def to_dict(self):
        '''Put the bubble parameters, simulation parameters, and results of 
        the simulation in a dict. Save just the names of the velocity field
        and the equation of motion (since these classes can't be pickled
        reliably)
        '''
        
        save_vars = ['bubble_params','sim_params',
                     'g_dir',
                     'x','v','u','dudt','velgrad',
                     'ti']
        res = {attr:getattr(self,attr) for attr in save_vars}
        res['velocity_field_name'] = self.velocity_field.name
        res['equation_of_motion_name'] = self.eom.name
        return res
    
    def save(self,fpath):
        '''put the results into a dict and pickle it'''
        res = self.to_dict()
        data.save_obj(res,fpath)
            
    def add_data_from_dict(self,res):
        
        # load the results if res is a filepath to them
        res = data.load_or_pass_on(res)
        
        # set the dict entries as attributes
        [setattr(self,key,res[key]) for key in res]
        
        # just to be safe, re-do the last timestep
        self.ti = max(0,self.ti-1)
        
'''
Functions involved in the equations of motion
'''

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
    drag = -1/8 * Cd * np.pi * d**2 * (slip.T*np.linalg.norm(slip,axis=-1)).T
    return drag

def calc_lift_force(slip,vort,d,Cl):
    lift = -1 * Cl * np.cross(slip,vort) * (d/2)**3*4./3*np.pi
    return lift

def a_bubble_MR_constantcoefficients(u,v,velgrad,dudt,d,Cd,Cm,Cl,g,g_dir):
    '''
    calculate a bubble's accceleration given its velocity, the local water
    velocity, and the bubble size
    '''
    
    
    vort = get_vorticity(velgrad)
    slip = v - u
    slip_drag = slip
    
    # pressure force
    press = calc_pressure_force(u,velgrad,dudt,d,Cm)
    
    # bouyant force
    grav = calc_grav_force(g,d,g_dir)
    
    # drag force    
    drag = calc_drag_force(slip_drag,d,Cd)
    
    # lift force
    lift = calc_lift_force(slip,vort,d,Cl)
    
    # calculate the added mass and the bubble acceleration
    m_added = Cm*(d/2)**3*4./3*np.pi
    a = (press+drag+grav+lift)/m_added
    
    return a

def calc_Cd_Snyder(Re):
    '''drag coefficient used in Snyder2007
    '''
    Re = np.atleast_1d(Re)
    Cd = np.zeros_like(Re)
    ix_low = np.argwhere(Re<1)
    ix_med = np.argwhere((Re>=1)*(Re<20))
    ix_high = np.argwhere(Re>=20)
    Cd[ix_low] = 24/Re[ix_low]
    Cd[ix_med] = (24./Re[ix_med]) * (1 + (3.6/Re[ix_med]**0.313)*((Re[ix_med]-1)/19)**2)
    Cd[ix_high] = (24./Re[ix_high]) * (1 + 0.15*Re[ix_high]**0.687)
    return Cd

def quiescent_speed(d,g,Cd):
    return np.sqrt(4./3 * d * g /Cd)

def bubble_params_given_nondim(Fr,dstar,u_vf,L_vf):
    d = dstar * L_vf
    g = u_vf**2 / (Fr**2*d)
    return d,g

def nu_given_Req(d,g,Cd_q,Re_q):
    '''calculate viscosity given the quiescent parameters
    '''
    v_q = quiescent_speed(d,g,Cd_q)
    nu = d*v_q / Re_q
    return nu

# class PointBubbleSimulation:
    
#     def __init__(self,params,fname_save=None):
        
#         self.params = params
        
#         # bubble parameters
#         self.beta = params['beta']
#         self.A = params['A']
#         self.Cm = params['Cm']
#         self.Cl = params['Cl']
#         self.Cd = params['Cd']
#         self.g = (u_rms**2/L_int)/self.A
#         self.v_q = u_rms/self.beta
#         self.d = (3./4) * self.Cd * self.v_q**2 / self.g
        
#         self.pressure_term_coef = 1 # to be multiplied by the pressure force
#         self.lift_term_coef = 1 # to be multiplied by the lift force (after accounting for C_L)
            
#         # replace the pressure, lift scalings with specified values if they were given
#         if 'pressure_term_coef' in params:
#             self.pressure_term_coef = params['pressure_term_coef']
#         if 'lift_term_coef' in params:
#             self.lift_term_coef = params['lift_term_coef']            
                
#         # simulation parameters
#         self.n_bubs = params['n_bubs']
#         self.dt_factor = params['dt_factor']
#         self.dt_use = self.dt_factor*dt
        
#         if 't_min' in list(params.keys()):
#             self.t_min = params['t_min']
#         else:
#             self.t_min = 0
#         if 't_max' in list(params.keys()):
#             self.t_max = params['t_max']
#         else:
#             self.t_max = t_max_turbulence
            
#         self.t = np.arange(self.t_min,self.t_max,self.dt_use)
#         self.n_t = len(self.t)
        
#         if fname_save is None:
#             self.fname_save = 'res_beta'+'{:03.2f}'.format(self.beta)+'_A'+'{:06.5f}'.format(self.A)+'_Cm'+'{:03.2f}'.format(self.Cm)+'_Cl'+'{:03.2f}'.format(self.Cl)+'_Cd'+'{:03.2f}'.format(self.Cd)+'.pkl'
#         else:
#             self.fname_save = fname_save
            
#     def save(self,fname_save=None):
#         if fname_save is None:
#             fname_save = self.fname_save
        
#         save_vars = ['beta','A',
#                      'Cm','Cl','Cd',
#                      'pressure_term_coef','lift_term_coef',
#                      'g','v_q','d','g_dir',
#                      'n_bubs','dt_factor','dt_use','t','n_t',
#                      'x','v','u','dudt','velgrad','ti']
        
#         res = {attr:getattr(self,attr) for attr in save_vars}
#         print('Saving data to '+fname_save)
#         with open(data_dir+fname_save, 'wb') as handle:
#             pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        
#     def init_sim(self):
#         '''
#         Initialize the simulation
#         '''
        
#         n_t = self.n_t
#         n_bubs = self.n_bubs
#         print(n_bubs)
        
#         # gravity direction chosen randomly for each bubble
#         self.g_dir = np.array([Rotation.random(1).apply([0,0,1]) for _ in range(n_bubs)])[:,0,:] # gravity direction
#         #self.g_dir = np.array([np.array([0,0,1]) for _ in range(n_bubs)])#[:,0,:] # gravity direction
        
#         self.x = np.zeros((n_t,n_bubs,3))
#         self.u = np.zeros((n_t,n_bubs,3))
#         self.v = np.zeros((n_t,n_bubs,3))
#         self.velgrad = np.zeros((n_t,n_bubs,3,3))
#         self.dudt = np.zeros((n_t,n_bubs,3))
        
#         self.x[0,...] = np.random.uniform(low=0,high=2*np.pi,size=(n_bubs,3))
                
#         self.ti = 0
#         self.t = np.arange(0,t_max_turbulence,self.dt_use)
#         self.n_t = len(self.t)
        
#     def add_data(self,res):
#         '''
#         Add partially-complete simulation data stored in the dict res
#         '''
        
#         self.x = res['x']
#         self.u = res['u']
#         self.v = res['v']
#         self.velgrad = res['velgrad']
#         self.dudt = res['dudt']
#         self.ti = res['ti']-1
#         self.g_dir = res['g_dir']
        
#     def add_data_if_existing(self,fname_save=None):
#         if fname_save is None:
#             fname_save = self.fname_save
            
#         if os.path.isfile(data_dir+fname_save):
#             print('Loading data from '+data_dir+fname_save)
#             with open(data_dir+fname_save, 'rb') as handle:
#                 res = pickle.load(handle)
#             self.add_data(res)
#         else:
#             print('Did not find file '+str(data_dir+fname_save))
        
#     def _advance(self,ti,use_filter=False):
        
#         t = self.t[ti]
#         u = self.u
#         v = self.v
#         x = self.x
#         dudt = self.dudt
#         velgrad = self.velgrad
#         #filter_size = (self.d/2)/np.sqrt(5)
#         #filter_size = (self.d) / (4./3 * np.pi)**(1./3)
#         filter_size = None
        
#         # liquid velocity
#         #print(d,self.d)
#         if use_filter==False:
#             u[ti+1,...] = interface.get_velocity(t,x[ti,...])
#         else:
#             u[ti+1,...] = interface.get_velocity_filtered(t,x[ti,...],filter_size)
#         if ti==0:
#             u[0,...] = u[1,...]
            
#         # surface average of liquid velocity
#         #print('getting surface average of the velocity')
#         if use_filter:
#             d2_factor = (filter_size+dx*4)/filter_size
#             u_drag = interface.surface_average_of_velocity(t,x[ti,...],self.d,filter_size,d2_factor=d2_factor,avg1=u[ti+1,...])
#         else:
#             u_drag = None
#         #print('... done')
        
#         # time velocity gradient
        
#         if use_filter==False:
#             delta = 1e-4
#             u_deltat = interface.get_velocity(t+delta,x[ti,...])
#         else:
#             delta = 1e-2
#             u_deltat = interface.get_velocity_filtered(t+delta,x[ti,...],filter_size)
#         dudt[ti+1,...] = (u_deltat-u[ti+1,...])/delta # future velocity at this point minus current velocity at this point
        
#         # liquid velocity gradient
#         if use_filter==False:
#             velgrad[ti+1,...] = interface.get_velocity_gradient(t,x[ti,...])
#         else:
#             velgrad[ti+1,...] = interface.get_velocity_gradient_filtered(t,x[ti,...],filter_size)
        
#         # start the simulation with v = u + v_q
#         if ti==0:
#             v[0,...] = u[1,...]*0 + self.g_dir * self.v_q
        
#         # bubble acceleration and new velocity
#         a = a_bubble(u[ti+1,...],v[ti,...],velgrad[ti+1,...],dudt[ti+1,...],self.d,self.Cd,self.Cm,self.Cl,self.g,self.g_dir,self.pressure_term_coef,self.lift_term_coef,u_drag=u_drag)
#         v[ti+1,...] = v[ti,...]+a*self.dt_use
        
#         # new position
#         x[ti+1,...] = x[ti,...]+v[ti+1,...]*self.dt_use
        
#         # store the data
#         self.u = u
#         self.v = v
#         self.x = x
#         self.dudt = dudt
#         self.velgrad = velgrad
        
#     def run_model(self,save_every=500):
        
#         print('running the model')
        
#         while self.ti < self.n_t-1:            
            
#             # advance the simulation
#             t_start = time_pkg.time()
#             self._advance(self.ti)
#             print(self.fname_save+' : Timestep '+str(self.ti)+', time '+'{:06.4f}'.format(self.t[self.ti])+', took '+'{:01.4f}'.format(time_pkg.time()-t_start)+' s.')
#             self.ti = self.ti + 1
            
#             # save, if necessary
#             if (self.ti % save_every) == 0:
#                 self.save()
                
#         self.save()
        

                
# def load_sim_from_file(fpath):
#     with open(fpath, 'rb') as handle:
#         res = pickle.load(handle)
#     p = PointBubbleSimulation(res)
#     p.add_data(res)
#     return(p)


    
# '''
# Lagrangian trajectories
# '''
    
# class LagrangianTrajectories:
    
#     def __init__(self,fname_save,n_traj=500,dt_factor=0.5):
#         self.fname_save = fname_save
#         self.n_traj = n_traj
#         self.dt_factor = dt_factor
#         self.dt_use = self.dt_factor*dt
        
#         self.t = np.arange(0,t_max_turbulence,self.dt_use)
#         self.n_t = len(self.t)
        
#     def init_sim(self):
#         '''
#         Initialize the simulation
#         '''
        
#         n_t = self.n_t
#         n_traj = self.n_traj
        
#         # gravity direction chosen randomly for each trajectory
#         self.g_dir = np.array([Rotation.random(1).apply([0,0,1]) for _ in range(n_traj)])[:,0,:] # gravity direction
        
#         self.x = np.zeros((n_t,n_traj,3))
#         self.u = np.zeros((n_t,n_traj,3))
        
#         self.x[0,...] = np.random.uniform(low=0,high=2*np.pi,size=(n_traj,3))
                
#         self.ti = 0
        
#     def add_data(self,res):
#         '''
#         Add partially-complete simulation data stored in the dict res
#         '''
        
#         self.x = res['x']
#         self.u = res['u']
#         self.ti = res['ti']-1
#         self.g_dir = res['g_dir']
        
#     def add_data_if_existing(self,fname_save=None):
#         if fname_save is None:
#             fname_save = self.fname_save
            
#         if os.path.isfile(data_dir+fname_save):
#             print('Loading data from '+data_dir+fname_save)
#             with open(data_dir+fname_save, 'rb') as handle:
#                 res = pickle.load(handle)
#             self.add_data(res)
#         else:
#             print('Did not find file '+str(data_dir+fname_save))
            
#     def save(self,fname_save=None):
#         if fname_save is None:
#             fname_save = self.fname_save
        
#         save_vars = ['n_traj','dt_factor','dt_use','t','n_t','g_dir',
#                      'x','u','ti']
        
#         res = {attr:getattr(self,attr) for attr in save_vars}
#         print('Saving data to '+fname_save)
#         with open(data_dir+fname_save, 'wb') as handle:
#             pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
#     def _advance(self,ti):
        
#         t = self.t[ti]
#         u = self.u
#         x = self.x
        
#         # liquid velocity
#         u[ti+1,...] = interface.get_velocity(t,x[ti,...])
#         if ti==0:
#             u[0,...] = u[1,...]
        
#         # new position
#         x[ti+1,...] = x[ti,...]+u[ti+1,...]*self.dt_use
        
#         # store the data
#         self.u = u
#         self.x = x
        
#     def run_model(self,save_every=500):
        
#         print('running the model')
        
#         while self.ti < self.n_t-1:            
            
#             # advance the simulation
#             t_start = time_pkg.time()
#             self._advance(self.ti)
#             print(self.fname_save+' : Timestep '+str(self.ti)+', time '+'{:06.4f}'.format(self.t[self.ti])+', took '+'{:01.4f}'.format(time_pkg.time()-t_start)+' s.')
#             self.ti = self.ti + 1
            
#             # save, if necessary
#             if (self.ti % save_every) == 0:                
#                 self.save()
                
#         self.save()
        
# def get_lagrangian_trajectories():
    
#     m  = LagrangianTrajectories('lagrangian_trajectories_dtFactor0.5',n_traj=500,dt_factor=0.5)
#     m.init_sim()
#     m.add_data_if_existing()
#     m.run_model()
    
# def run_filt_sim(beta,d_star,fname_save=None,Fr=None):
    
#     '''
#     python3 -c "from point_bubble_JHTDB.model import run_filt_sim; run_filt_sim(1,0.12,fname_save=None)"
#     python3 -c "from point_bubble_JHTDB.model import run_filt_sim; run_filt_sim(1,None,Fr=0.2)"
#     '''
    
#     if fname_save is None and d_star is not None:
#         fname_save = 'filt_beta'+'{:01.2f}'.format(beta)+'_dstar'+'{:01.3f}'.format(d_star)+'_filtSizeForSphereVol.pkl'
    
#     params = default_params 
    
#     Cd = 1
    
#     if Fr is None:
#         A = A_given_dByL(d_star,beta,Cd)
        
#     else:
#         A = Fr**2
#         fname_save = 'filt_beta'+'{:01.2f}'.format(beta)+'_Fr'+'{:01.3f}'.format(Fr)+'_filtSizeForSphereVol.pkl'
#     #print(beta)
#     #print(d_star)
#     #else:
#         #A = A
#         #d_star = 
#     params['beta'] = beta
#     params['A'] = A
#     params['Cd'] = Cd
#     params['Cl'] = 0
#     params['n_bubs'] = 500
#     p = PointBubbleSimulation(params,fname_save=fname_save)
#     p.init_sim()
#     p.add_data_if_existing()
#     p.run_model()