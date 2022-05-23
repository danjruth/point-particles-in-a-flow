# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 15:11:13 2021

@author: druth
"""

import numpy as np
from pointparticlesinaflow import EquationOfMotion, Force
import scipy.optimize

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
        p['slip'] = p['v']-p['u']
        return p
    
class MaxeyRileyPointBubbleSnyder2007(EquationOfMotion):
    '''Maxey-Riley equation for a bubble in a much denser liquid, as used in 
    Snyder2007 [not yet complete]
    '''
    def __init__(self):
        super().__init__(name='MaxeyRiley_pointbubble_constantcoefficients',
                         forces=[PressureForceBubble(),
                                 GravForceBubble(),
                                 DragForceSnyder2007(),
                                 ConstantCLLiftForce()])
        
    def calc_m_eff(self,p):
        return p['Cm']*(p['d']/2)**3*4./3*np.pi
        
    def _pre_calculations(self,p):
        p['vort'] = get_vorticity(p['velgrad'])
        p['slip'] = p['v']-p['u']
        return p
    
class MaxeyRileyPointBubbleConstantCoefsVisc(EquationOfMotion):
    '''Maxey-Riley equation for a bubble in a much denser liquid, with constant
    lift and added-mass coefficients, and Cd=24./Re for linear drag.
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
        p['slip'] = p['v']-p['u']
        return p
    
class LagrangianEOM(EquationOfMotion):
    def __init__(self):
        super().__init__(name='Lagrangian')
        
    def __call__(self,p,dt):
        '''overwrite to return the fluid velocity at the particle locations;
        otherwise it would try to sum forces, but there are no forces for this
        EOM
        '''
        return p['u']
    
'''
Helper, pre-calculation functions
'''
def _calc_slip(r):
    r['slip'] = r['v']-r['u']
    return r

def _calc_vort(r):
    r['vort'] = get_vorticity(r['velgrad'])
    return r

'''
Forces
'''

class PressureForceBubble(Force):    
    def __init__(self):
        super().__init__(name='pressure_bubble',short_name='press')
        self.pkeys = ['d','Cm']
    def __call__(self,p):
        u = p['u']
        velgrad = p['velgrad']
        dudt = p['dudt']
        d = p['d']
        Cm = p['Cm']
        u_times_deldotu = np.array([np.sum(velgrad[...,i,:]*u,axis=-1) for i in range(3)])
        u_times_deldotu = np.moveaxis(u_times_deldotu,0,-1)
        #print(np.shape(u_times_deldotu))
        #print(np.shape(dudt))
        #print(np.shape(d))
        press = (1+Cm) * (d/2)**3*4./3*np.pi * np.moveaxis((dudt + u_times_deldotu),-2,-1)
        press = np.moveaxis(press,-1,-2)
        return press

class GravForceBubble(Force):    
    def __init__(self):
        super().__init__(name='grav_bubble',short_name='grav')
        self.pkeys = ['d','g_dir','g']
    def __call__(self,p):
        g = p['g']
        d = p['d']
        g_dir = p['g_dir']
        grav = -1 * g*(d/2)**3*4./3*np.pi * np.moveaxis(g_dir,0,-1)
        return np.moveaxis(grav,-1,0)

class ConstantCDDragForce(Force):    
    def __init__(self):
        super().__init__(name='constant_CD_drag',short_name='drag')
        self.pkeys = ['d','Cd']
        self.precalcs = [_calc_slip]
    def __call__(self,r):
        Cd = r['Cd']
        d = r['d']
        slip = r['slip'] # [n_particles,3]        
        slip_mag = np.linalg.norm(slip,axis=-1)
        slip = np.moveaxis(slip,-1,0)
        drag = -1/8 * Cd * np.pi * d**2 * (slip*slip_mag)
        drag = np.moveaxis(drag,0,-1)
        return drag
    
class DragForceSnyder2007(Force):    
    def __init__(self):
        super().__init__(name='drag_Snyder2007',short_name='drag')
        self.pkeys = ['d','nu']
        self.precalcs = [_calc_slip]
    def __call__(self,p):
        d = p['d']
        nu = p['nu']
        slip = p['slip']
        Re = np.linalg.norm(slip,axis=-1)*d/nu
        Cd = calc_Cd_Snyder(Re)
        drag = calc_drag_force(slip,d,Cd)
        return drag

class ViscousDragForce(Force):
    def __init__(self):
        super().__init__(name='viscous_drag',short_name='drag')
        self.pkeys = ['d','nu']
        self.precalcs = [_calc_slip]
    def __call__(self,p):
        nu = p['nu']
        d = p['d']
        slip = p['slip']
        # slip_norm = np.linalg.norm(slip,axis=-1)
        # Re = slip_norm * d / nu # [n_bub] or [n_t,n_bub]
        # Re[Re==0] = 1 # doesn't matter so avoid divide by 0
        # Cd = 24./Re
        # Cd = np.moveaxis(np.array([Cd]*3),0,-1) # [n_bub,3] or [n_t,n_bub,3]
        #drag = calc_drag_force(slip,d,Cd)
        drag = -3 * np.pi * nu * d * np.moveaxis(slip,-2,-1)
        drag = np.moveaxis(drag,-1,-2)
        return drag

class ConstantCLLiftForce(Force):    
    def __init__(self):
        super().__init__(name='lift_bubble',short_name='lift')   
        self.pkeys = ['d','Cl']
        self.precalcs = [_calc_slip,_calc_vort]
    def __call__(self,r):
        Cl = r['Cl']
        slip = r['slip']
        vort = r['vort']
        d = r['d']
        lift = -1 * Cl * np.moveaxis(np.cross(slip,vort),-2,-1) * (d/2)**3*4./3*np.pi
        lift = np.moveaxis(lift,-1,-2)
        return lift
    
'''
Helper functions for calculating forces
'''

def calc_drag_force(slip,d,Cd):
    # need to move component axis of slip, Cd to front for multiplication
    slip_mag = np.linalg.norm(slip,axis=-1)
    slip = np.moveaxis(slip,-1,0)
    if len(np.shape(Cd))>1:
        Cd = np.moveaxis(Cd,-1,0)
    #print(np.shape(Cd))
    #print(np.shape(slip))
    #print(np.shape(slip_mag))
    drag = -1/8 * Cd * np.pi * d**2 * (slip*slip_mag)
    drag = np.moveaxis(drag,0,-1)
    return drag

def get_vorticity(velgrad):
    shape_velgrad = np.shape(velgrad)
    vort = np.zeros(shape_velgrad[:-1]) # 
    vort[...,0] = velgrad[...,2,1] - velgrad[...,1,2]
    vort[...,1] = velgrad[...,0,2] - velgrad[...,2,0]
    vort[...,2] = velgrad[...,1,0] - velgrad[...,0,1]
    return vort

def calc_Cd_Snyder(Re):
    '''drag coefficient used in Snyder2007
    '''
    Re = np.atleast_1d(Re)
    Cd = np.zeros_like(Re)
    # ix_low = np.argwhere(Re<1)
    # ix_med = np.argwhere((Re>=1)*(Re<20))
    # ix_high = np.argwhere(Re>=20)
    # ix_0 = np.argwhere(Re==0)
    # Cd[ix_low] = 24/Re[ix_low]
    # Cd[ix_med] = (24./Re[ix_med]) * (1 + (3.6/Re[ix_med]**0.313)*((Re[ix_med]-1)/19)**2)
    # Cd[ix_high] = (24./Re[ix_high]) * (1 + 0.15*Re[ix_high]**0.687)
    # Cd[ix_0] = 1 # value doesn't matter, just can't be nan or inf
    Cd[Re<1] = 24/Re[Re<1]
    Cd[Re>=1] = (24./Re[Re>=1]) * (1 + (3.6/Re[Re>=1]**0.313)*((Re[Re>=1]-1)/19)**2)
    Cd[Re>=20] = (24./Re[Re>=20]) * (1 + 0.15*Re[Re>=20]**0.687)
    Cd[Re==0] = 1 # value doesn't matter, just can't be nan or inf
    
    # give it a "component" axis
    #Cd = np.moveaxis([Cd]*3,0,-1) # [bub,component] or [time,bub,component]
    
    return Cd

def calc_vq_Snyder(d,nu,g,):
    buoyancy_force = g * np.pi * (d/2)**3 * (4./3)
    def calc_resid(v_q):            
        Re = v_q*d/nu
        Cd = calc_Cd_Snyder(Re)[0]        
        drag_force = calc_drag_force(np.array([0,0,v_q]),d,Cd)[-1] * -1
        return np.abs(np.mean((drag_force-buoyancy_force)))
    res = scipy.optimize.minimize(calc_resid,x0=[0.5,],tol=buoyancy_force/100000,method='Powell',bounds=[(0,np.inf)]) # ,tol=1e-2
    assert res.success
    v_q = res.x[0]
    return v_q

'''
Other functions
'''
