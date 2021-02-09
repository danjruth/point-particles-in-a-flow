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
    
class LagrangianEOM(EquationOfMotion):
    def __init__(self):
        super().__init__(name='Lagrangian')
        
    def __call__(self,p,dt):
        '''return the fluid velocity at the particle locations
        '''
        return p['u']
    
'''
Forces
'''

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

class GravForceBubble(Force):    
    def __init__(self):
        super().__init__(name='grav_bubble')
        self.const_params = ['d','g_dir','g']
    def __call__(self,p):
        g = p['g']
        d = p['d']
        g_dir = p['g_dir']
        return g*(d/2)**3*4./3*np.pi * g_dir

class ConstantCDDragForce(Force):    
    def __init__(self):
        super().__init__(name='constant_CD_drag')
        self.const_params = ['d','Cd']
    def __call__(self,p):
        Cd = p['Cd']
        d = p['d']
        slip = p['slip']
        return calc_drag_force(slip,d,Cd)

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
    
'''
Helper functions for calculating forces
'''

def calc_drag_force(slip,d,Cd):
    drag = -1/8 * Cd * np.pi * d**2 * (slip.T*np.linalg.norm(slip,axis=-1)).T
    return drag

def get_vorticity(velgrad):
    vort = np.zeros((len(velgrad),3)) # 
    vort[:,0] = velgrad[...,2,1] - velgrad[...,1,2]
    vort[:,1] = velgrad[...,0,2] - velgrad[...,2,0]
    vort[:,2] = velgrad[...,1,0] - velgrad[...,0,1]
    return vort

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