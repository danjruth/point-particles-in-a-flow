# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 15:11:13 2021

@author: druth
"""

import numpy as np
from . import EquationOfMotion, Force

'''
Classes for equations of motion
'''
 
class MaxeyRileyPointBubbleConstantCoefs(EquationOfMotion):
    '''Maxey-Riley equation for a bubble in a much denser liquid, with constant
    lift, drag, and added-mass coefficients
    '''
    def __init__(self):
        super().__init__(name='MaxeyRiley_pointbubble_constantcoefficients',
                         forces=[PressureForceBubble(),
                                 GravForceBubble(),
                                 ConstantCDDragForce(),
                                 ConstantCLLiftForce()])
        
    def calc_m_eff(self,p):
        return p['Cm']*(p['d']/2)**3*4./3*np.pi
        
    def _pre_calculations(self,p):
        p['vort'] = get_vorticity(p['velgrad'])
        return p
    
class MaxeyRileyPointBubbleConstantCoefsVisc(EquationOfMotion):
    '''Maxey-Riley equation for a bubble in a much denser liquid, with constant
    lift, drag, and added-mass coefficients
    '''
    def __init__(self):
        super().__init__(name='MaxeyRiley_pointbubble_constantcoefficients',
                         forces=[PressureForceBubble(),
                                 GravForceBubble(),
                                 ViscousDragForce(),
                                 ConstantCLLiftForce()])
        
    def calc_m_eff(self,p):
        return p['Cm']*(p['d']/2)**3*4./3*np.pi
        
    def _pre_calculations(self,p):
        p['vort'] = get_vorticity(p['velgrad'])
        return p
    
# class MaxeyRileyPointBubbleConstantCoefsVisc(EquationOfMotion):
#     '''Maxey-Riley equation for a bubble in a much denser liquid, with constant
#     lift and added-mass coefficients and linear drag set by C_D = 24/Re
#     '''
#     def __init__(self):
#         super().__init__(name='MaxeyRiley_pointbubble_constantcoefficients_viscous')
        
#     def __call__(self,v,fs,sim,dt):
#         '''calculate a new v based on the current v, the field state, and the
#         bubble parameters stored in sim'''
#         a = a_bubble_MR_constantcoefficients_visc(fs.u,v,fs.velgrad,fs.dudt,sim.d,sim.nu,sim.Cm,sim.Cl,sim.g,sim.g_dir)
#         return v+a*dt
    
class MaxeyRileyPointBubbleVariableCoefs(EquationOfMotion):
    '''Maxey-Riley equation for a bubble in a much denser liquid, with constant
    lift and added-mass coefficients and variable drag, as used in Snyder2007
    '''
    def __init__(self):
        super().__init__(self,name='MaxeyRiley_pointbubble_variablecoefficients')
                
    def __call__(self,v,fs,sim,dt):
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
        super().__init__(self,name='Lagrangian')
        
    def __call__(self,v,fs,sim,dt):
        '''return the fluid velocity at the particle locations
        '''
        return fs.u
    
'''
Functions used in the equations of motion
'''

def get_vorticity(velgrad):
    vort = np.zeros((len(velgrad),3)) # 
    vort[:,0] = velgrad[...,2,1] - velgrad[...,1,2]
    vort[:,1] = velgrad[...,0,2] - velgrad[...,2,0]
    vort[:,2] = velgrad[...,1,0] - velgrad[...,0,1]
    return vort

class PressureForceBubble(Force):    
    def __init__(self):
        super().__init__(name='pressure_bubble')
        self.const_params = ['d','Cm']
    def __call__(self,p):
        u = p['u']
        velgrad = p['velgrad']
        dudt = p['dudt']
        d = p['d']
        Cm = p['Cm']
        u_times_deldotu = np.array([np.sum(velgrad[...,0,:]*u,axis=-1),
                                np.sum(velgrad[...,1,:]*u,axis=-1),
                                np.sum(velgrad[...,2,:]*u,axis=-1)]).T
        press = (1+Cm) * (d/2)**3*4./3*np.pi * (dudt + u_times_deldotu)
        return press

def calc_pressure_force(u,velgrad,dudt,d,Cm):
    # pressure force and added mass, in terms of the carrier velocity field
    u_times_deldotu = np.array([np.sum(velgrad[...,0,:]*u,axis=-1),
                                np.sum(velgrad[...,1,:]*u,axis=-1),
                                np.sum(velgrad[...,2,:]*u,axis=-1)]).T
    press = (1+Cm) * (d/2)**3*4./3*np.pi * (dudt + u_times_deldotu)
    return press

class GravForceBubble(Force):    
    def __init__(self):
        super().__init__(name='grav_bubble')
        self.const_params = ['d','g_dir','g']
    def __call__(self,p):
        g = p['g']
        d = p['d']
        g_dir = p['g_dir']
        return g*(d/2)**3*4./3*np.pi * g_dir

def calc_grav_force(g,d,g_dir):
    return g*(d/2)**3*4./3*np.pi * g_dir

class ConstantCDDragForce(Force):    
    def __init__(self):
        super().__init__(name='constant_CD_drag')
        self.const_params = ['d','Cd']
    def __call__(self,p):
        Cd = p['Cd']
        d = p['d']
        slip = p['slip']
        return -1/8 * Cd * np.pi * d**2 * (slip.T*np.linalg.norm(slip,axis=-1)).T

def calc_drag_force(slip,d,Cd):
    drag = -1/8 * Cd * np.pi * d**2 * (slip.T*np.linalg.norm(slip,axis=-1)).T
    return drag

class ViscousDragForce(Force):    
    def __init__(self):
        super().__init__(name='viscous_drag')
        self.const_params = ['d','nu']
    def __call__(self,p):
        nu = p['nu']
        d = p['d']
        slip = p['slip']
        slip_norm = np.linalg.norm(slip,axis=-1)
        Re = slip_norm * d / nu
        Cd = 24./Re
        Cd = np.array([Cd]*3).T
        return calc_drag_force(slip,d,Cd)

def calc_drag_force_visc(slip,d,nu):
    slip_norm = np.linalg.norm(slip,axis=-1)
    Re = slip_norm * d / nu
    Cd = 24./Re
    Cd = np.array([Cd]*3).T
    return calc_drag_force(slip,d,Cd)

class ConstantCLLiftForce(Force):    
    def __init__(self):
        super().__init__(name='lift_bubble')     
        self.const_params = ['d','Cl']
    def __call__(self,p):
        Cl = p['Cl']
        slip = p['slip']
        vort = p['vort']
        d = p['d']
        return -1 * Cl * np.cross(slip,vort) * (d/2)**3*4./3*np.pi

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

def a_bubble_MR_constantcoefficients_visc(u,v,velgrad,dudt,d,nu,Cm,Cl,g,g_dir):
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
    drag = calc_drag_force_visc(slip_drag,d,nu)
    
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