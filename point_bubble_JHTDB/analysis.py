# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 11:55:39 2020

@author: ldeike
"""

import numpy as np
import pickle
from point_bubble_JHTDB import model
import pandas as pd

def get_hist(y,bins=1001):
    '''return a normalized pdf and x locs of bin centers'''
    hist,edges = np.histogram(y[~np.isnan(y)],bins=bins,density=True)
    return edges[:-1]+np.diff(edges)/2, hist

def get_rot_dirs(g_dir):
    '''get vectors denoting the new x,y,z directions wrt the DNS coordinate system'''
    
    # z is in the direction of gravity
    z_dir = g_dir.copy()
    
    # x direction has to be normal to z, and we arbitrarily choose it's also normal to the DNS x
    x_dir_unscaled = np.cross(z_dir,[1,0,0])
    x_dir = x_dir_unscaled / np.linalg.norm(x_dir_unscaled)
    
    # get the y direction
    y_dir = np.cross(z_dir,x_dir)
    

def rot_coord_system(arr,g_dir):
    '''
    coordinate system rotation to align z with gravity, just for a single bubble.
    '''

    # z is in the direction of gravity
    z_dir = g_dir.copy()
    
    # x direction has to be normal to z, and we arbitrarily choose it's also normal to the DNS x
    x_dir_unscaled = np.cross(z_dir,[1,0,0])
    x_dir = x_dir_unscaled / np.linalg.norm(x_dir_unscaled)
    
    # get the y direction
    y_dir = np.cross(z_dir,x_dir)
    
    arr_rot = np.array([np.dot(arr,x_dir),np.dot(arr,y_dir),np.dot(arr,z_dir)]).T

    return arr_rot

def rot_all(arrs,g_dirs,actually_rot=True):    
    '''
    rotate the coordinate systems for all n_b bubbles (along axis 1). 
    
    arrs has shape (n_t,n_b,3); g_dirs has shape (n_b,3)
    '''
    
    if actually_rot:
        arrs_new = []
        for i in np.arange(len(g_dirs)):
            arrs_new.append(rot_coord_system(arrs[:,i,:],g_dirs[i,:]))        
        arrs_new = np.moveaxis(np.array(arrs_new),0,1)
        return arrs_new
    else:
        return arrs

def load_case(d,n_T_int=2,rot=True):
    
    if isinstance(d,str):
        with open(d, 'rb') as handle:
            res = pickle.load(handle)
    else:
        res = d
        
    res['Fr'] = model.u_rms / np.sqrt(res['g']*model.L_int)
        
    # calculate the forces, in DNS coords initially
    slip = res['v'][:-1] - res['u'][1:]
    vort = get_vorticity(res['velgrad'])
    vol = (res['d']/2.)**3 * 4./3 * np.pi
    
    grav_z = res['g'] * vol
    norm = grav_z
    
    press = []
    drag = []
    lift = []
    for i in [0,1,2]:
        
        u_times_deldotu = np.sum(res['velgrad'][1:,:,i,:]*res['u'][1:],axis=-1)
        press.append((1+res['Cm'])* vol * (res['dudt'][1:,:,i] + u_times_deldotu))        
        drag.append(-1*res['Cd'] * 0.5 * np.pi * (res['d']/2)**2 * slip[...,i] * np.linalg.norm(slip[:,...],axis=-1))        
        lift.append(-1 * res['Cl'] * np.cross(slip,vort[1:])[...,i] * vol)
        
    # make each arrays, and rotate
    press = np.moveaxis(np.array(press),0,-1)
    drag = np.moveaxis(np.array(drag),0,-1)
    lift = np.moveaxis(np.array(lift),0,-1)
    res['press'] = rot_all(press,res['g_dir'],actually_rot=rot)
    res['drag'] = rot_all(drag,res['g_dir'],actually_rot=rot)
    res['lift'] = rot_all(lift,res['g_dir'],actually_rot=rot)
    res['grav_z'] = grav_z
        
    # rotate the velocities and position
    res['v'] = rot_all(res['v'],res['g_dir'],actually_rot=rot)
    res['u'] = rot_all(res['u'],res['g_dir'],actually_rot=rot)
    res['x'] = rot_all(res['x'],res['g_dir'],actually_rot=rot)
    res['slip'] = rot_all(slip,res['g_dir'],actually_rot=rot)
    res['vort'] = rot_all(vort,res['g_dir'],actually_rot=rot)
    res['dudt'] = rot_all(res['dudt'],res['g_dir'],actually_rot=rot)
    
    # drop the velgrad since it hasn't been rotated
    del res['velgrad']
    
    # see which points to consider (after initial transient)
    res['cond'] = res['t']>model.T_int*n_T_int
    mean_rise = np.mean(res['v'][:,:,2],axis=1)
    res['cond'] = res['cond'] * (mean_rise>0)
    
    # get rid of the final point in time, so everything has the same length (slip already is the right length)
    for var in ['v','u','x','vort','t','cond']:
        res[var] = res[var][:-1]
    
    return res

def concat_cases(list_of_dicts, fpath_save=None):
    
    # start with making a copy of the first dict
    res = list_of_dicts[0].copy()
    
    # sum the total number of bubbles
    res['n_bubs'] = np.sum([d['n_bubs'] for d in list_of_dicts])
    
    # no time axis for g_dir, so concat along the 0th axis
    res['g_dir'] = np.concatenate([d['g_dir'] for d in list_of_dicts],axis=0)
    
    # bubble axis is 1 for the rest of the variables to concatenate
    keys_to_concat = ['x','v','u','dudt','velgrad']
    for key in keys_to_concat:
        res[key] = np.concatenate([d[key] for d in list_of_dicts],axis=1)
        
    if fpath_save is not None:
        print('Saving data to '+fpath_save)
        with open(fpath_save, 'wb') as handle:
            pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    return res
        

def get_vorticity(velgrad):
    # similar to the function in model.py
    velgrad_shape = np.shape(velgrad)
    vort_shape = velgrad_shape[:-1]
    vort = np.zeros(vort_shape)
    vort[...,0] = velgrad[...,2,1] - velgrad[...,1,2]
    vort[...,1] = velgrad[...,0,2] - velgrad[...,2,0]
    vort[...,2] = velgrad[...,1,0] - velgrad[...,0,1]
    return vort

def get_curvature(vel,t):
    accel = (np.gradient(vel,axis=0).T/np.gradient(t)).T
    curvature = np.linalg.norm(np.cross(vel,accel),axis=-1) / np.linalg.norm(vel,axis=-1)**3
    return curvature

def get_powerlaw(x,y,roll_window=1):
    powerlaw = np.gradient(np.log(y))/np.gradient(np.log(x))
    powerlaw = pd.Series(data=powerlaw).rolling(center=True,window=roll_window,min_periods=0).mean()
    return powerlaw

def get_minmax_series(df,varx,vary):

    x = df[varx].unique()
        
    low = np.zeros_like(x)
    high = np.zeros_like(x)
    
    for bi,b in enumerate(x):
        low[bi] = df[df[varx]==b][vary].min()
        high[bi] = df[df[varx]==b][vary].max()
        
    return x,low,high
    
