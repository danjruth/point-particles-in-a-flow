# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 16:50:22 2020

@author: danjr
"""

import numpy as np

try:
    from pyJHTDB import libJHTDB
    lJHTDB = libJHTDB()
    lJHTDB.initialize(exit_on_error=True) # hopefully this will throw an error instead of just exiting?
except ImportError:
    print('Unable to import pyJHTDB and initialize')


def get_velocity(t,x,lJHTDB=lJHTDB):
    #print(x.astype(np.float32))
    return lJHTDB.getData(t, point_coords=x.copy().astype(np.float32), data_set='isotropic1024coarse', getFunction='getVelocity', sinterp='Lag4', tinterp='PCHIPInt')

def get_velocity_gradient(t,x,lJHTDB=lJHTDB):
    
    # query the data
    result_grad = lJHTDB.getData(t, x.copy().astype(np.float32), data_set='isotropic1024coarse',sinterp='FD4Lag4', tinterp='PCHIPInt',getFunction='getVelocityGradient')
    
    # put it into the correct shape
    result_grad_new = np.zeros((len(x),3,3)).astype(np.float32)
    for i in range(len(x)):
        this_velgrad = np.array([result_grad[i,0:3],result_grad[i,3:6],result_grad[i,6:]])
        result_grad_new[i,...] = this_velgrad
        
    return result_grad_new

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

    u_flat = get_velocity(t,points_flat,lJHTDB=lJHTDB)
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