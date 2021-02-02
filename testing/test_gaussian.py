# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 11:49:32 2021

@author: druth
"""

from point_bubble_JHTDB import model, analysis
import numpy as np
import matplotlib.pyplot as plt

'''
Construct the velocity field
'''
vf = model.VelocityField()


n_modes = 8
u_rms = 0.1
L_int = 0.01
T_int = L_int/u_rms

mr = model.MREqn()

bubble_params = {'d':2e-3,
                 'g':9.81,
                 'Cm':0.5,
                 'Cd':0.5,
                 'Cl':0.0}

sim_params = {'n_bubs':80,
              'dt':1e-4,
              't_min':0,
              't_max':0.4,
              'fname':'test'}

def make_vf(n_modes=8,u_rms=0.1,L_int=0.01):

    T_int = L_int/u_rms
    
    b = np.random.normal(scale=u_rms,size=(n_modes,3))
    c = np.random.normal(scale=u_rms,size=(n_modes,3))
    k = np.random.normal(scale=1./L_int,size=(n_modes,3))
    omega = np.random.normal(scale=1/T_int,size=(n_modes))*0
    
    def get_velocity(t,x):
        vel = np.zeros((len(x),3))
        for m in range(n_modes):
            # outer product of the sin/cos term (len n_bubs) and the 3 coefficients for this mode gives shape (n_bubs,3)
            mode_contribution = np.outer(np.sin(np.dot(x,k[m,:])+omega[m]*t),b[m,:]) + np.outer(np.cos(np.dot(x,k[m,:])+omega[m]*t),c[m,:])
            vel = vel + mode_contribution
        return vel/np.sqrt(n_modes)
    
    def get_velocity_gradient(t,x):
        velgrad = np.zeros((len(x),3,3))
        for m in range(n_modes):
            mode_contribution = np.zeros((len(x),3,3))
            for j in range(3):
                mode_contribution[:,:,j] = k[m,j]*np.outer(np.cos(np.dot(x,k[m,:])+omega[m]*t),b[m,:]) - k[m,j]*np.outer(np.sin(np.dot(x,k[m,:])+omega[m]*t),c[m,:])
            velgrad = velgrad + mode_contribution
        return velgrad/np.sqrt(n_modes)
    
    def get_dudt(t,x,u_t=None):
        dudt = np.zeros((len(x),3))
        for m in range(n_modes):
            mode_contribution = omega[m]*np.outer(np.cos(np.dot(x,k[m,:])+omega[m]*t),b[m,:]) - omega[m]*np.outer(np.sin(np.dot(x,k[m,:])+omega[m]*t),c[m,:])
            dudt = dudt + mode_contribution
        return dudt/np.sqrt(n_modes)
    
    vf.get_velocity = get_velocity
    vf.get_velocity_gradient = get_velocity_gradient
    vf.get_dudt = get_dudt
    
    return vf

n_mode_vec = [2,4,6,8,]*6
mean_speed_vec = []
for n_modes in n_mode_vec:

    vf = make_vf(n_modes=n_modes,u_rms=u_rms,L_int=L_int)
    
    sim = model.Simulation(vf,bubble_params,sim_params,mr)
    sim.init_sim()
    
    for ti in np.arange(0,sim.n_t-1,1):
        sim._advance(ti)
        
    res = sim.save()
    
    v = analysis.rot_all(sim.v,sim.g_dir,)
    u = analysis.rot_all(sim.u,sim.g_dir,)
    x = analysis.rot_all(sim.x,sim.g_dir,)
    
    v_q = model.quiescent_speed(bubble_params['d'],bubble_params['g'],bubble_params['Cd'])
    
    fig,ax = plt.subplots()
    for i in np.arange(np.shape(v)[1]):
        ax.plot(sim.t/T_int,v[:,i,2]/v_q,lw=2,alpha=0.6)
    ax.plot(sim.t/T_int,v[:,:,2].mean(axis=1)/v_q,lw=2,alpha=1,color='k')
    mean_speed_vec.append(v[...,2].mean()/v_q)
    

#return v[...,2].mean()/v_q

#n_mode_vec = [16,24,48]*4
#mean_speed_vec = [get_mean_speed(n_modes) for n_modes in n_mode_vec]


plt.figure()
plt.plot(n_mode_vec,mean_speed_vec,'o')
msv = np.array(mean_speed_vec)
plt.plot(np.unique(n_mode_vec),[np.mean(msv[n_mode_vec==nm]) for nm in np.unique(n_mode_vec)],'-')

stophere


print('beta = '+str(u_rms/v_q))
print('d_star = '+str(bubble_params['d']/L_int))
print('n_modes = '+str(n_modes))
print('nondim speed = '+str(v[:,:,2].mean()/v_q))

# image of the field
x = np.linspace(0,0.2,201)
y = np.linspace(0,0.2,101)
X,Y = np.meshgrid(x,y)
Z = np.zeros_like(X)
xyz_flat = np.array([X.flatten(),Y.flatten(),Z.flatten()]).T
vel_flat = vf.get_velocity(0,xyz_flat)
vel = np.reshape(vel_flat,(np.shape(X)[0],np.shape(X)[1],3))
fig,axs = plt.subplots(1,3,figsize=(12,4))
for i,ax in enumerate(axs):
    ax.pcolormesh(X,Y,vel[:,:,i],cmap='seismic')
    ax.set_aspect('equal')
    
    
# image of the velocity gradient
x = np.linspace(0,0.2,201)
y = np.linspace(0,0.2,101)
X,Y = np.meshgrid(x,y)
Z = np.zeros_like(X)
xyz_flat = np.array([X.flatten(),Y.flatten(),Z.flatten()]).T
velgrad_u_flat = vf.get_velocity_gradient(0,xyz_flat)[...,0]
velgrad_u = np.reshape(velgrad_u_flat,(np.shape(X)[0],np.shape(X)[1],3))
fig,axs = plt.subplots(1,3,figsize=(12,4))
for i,ax in enumerate(axs):
    ax.pcolormesh(X,Y,velgrad_u[:,:,i],cmap='seismic')
    ax.set_aspect('equal')
    
    
x = np.linspace(0,1,10001)
z = np.zeros_like(x)
y = np.zeros_like(x)
vel = vf.get_velocity(0,np.array([x,y,z]).T)
plt.figure()
plt.plot(x,vel)

n_mode_vec = np.array([3,12,24,48,64])
u_rms_vec = []
for n_modes in n_mode_vec:
    print(n_modes)
    n_f = 8
    velfields = []
    for fi in range(n_f):
        vf = make_vf(n_modes=n_modes)
        velfields.append(vf.get_velocity(0,np.array([x,y,z]).T))
    velfields = np.array(velfields)
    u_rms_vec.append(np.std(velfields))
plt.figure()
plt.loglog(n_mode_vec,np.array(u_rms_vec),'o')
plt.axhline(u_rms)
#plt.loglog(n_mode_vec,n_mode_vec**0.5*u_rms)