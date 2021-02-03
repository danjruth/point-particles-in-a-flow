# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 11:49:32 2021

@author: druth
"""

from point_bubble_JHTDB import model, analysis
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''
Construct the velocity field
'''
vf = model.VelocityField()


n_modes = 8
u_rms = 0.229
L_int = 0.01
T_int = L_int/u_rms
import toolkit.parallel

mr = model.MREqn()

bubble_params = {'d':2e-3,
                 'g':9.81,
                 'Cm':0.5,
                 'Cd':1,
                 'Cl':0.0}

sim_params = {'n_bubs':200,
              'dt':1e-4,
              't_min':0,
              't_max':0.4,
              'fname':'test'}

def make_vf(n_modes=8,u_rms=0.1,L_int=0.01):

    T_int = L_int/u_rms
    
    # random coefficients/wavenumbers/frequencies for each mode
    b = np.random.normal(scale=u_rms,size=(n_modes,3))
    c = np.random.normal(scale=u_rms,size=(n_modes,3))
    k = np.random.normal(scale=1./L_int,size=(n_modes,3))
    omega = np.random.normal(scale=1/T_int,size=(n_modes))
    
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

class Case:
    
    def __init__(self,d,g,L_int,u_rms):
        
        self.d = d
        self.g = g
        self.L_int = L_int
        self.u_rms = u_rms
        
        
        self.bubble_params = {'d':d,
                         'g':g,
                         'Cm':0.5,
                         'Cd':1,
                         'Cl':0.0}

        self.sim_params = {'n_bubs':200,
                      'dt':1e-4,
                      't_min':0,
                      't_max':max(L_int/u_rms*2,np.sqrt(L_int/g)*2,0.4),
                      'fname':'test'}
        
        self.v_q = model.quiescent_speed(d, g, self.bubble_params['Cd'])
        self.beta = self.u_rms / self.v_q
        self.Fr = self.u_rms / np.sqrt(self.d*self.g)
        self.d_star = self.d / self.L_int
        
    def run(self,n=20):
        
        def do_job(i):
            vf = make_vf(n_modes=8,u_rms=self.u_rms,L_int=self.L_int)
            sim = model.Simulation(vf,self.bubble_params,self.sim_params,mr)
            sim.init_sim()
            for ti in np.arange(0,sim.n_t-1,1):
                sim._advance(ti)
            v = analysis.rot_all(sim.v,sim.g_dir,)
            nondim_speed = np.mean(v[int(len(v)/2):,:,2]) / self.v_q
            return nondim_speed
        
        self.nondim_speeds = np.array(toolkit.parallel.parallelize_job(do_job, range(n)))
        self.nondim_speed = np.mean(self.nondim_speeds)
        
#c = Case(2e-3,9.81,0.01,0.1)
#c.run()
#stophere

#d_vec = [1e-3,2e-3,3e-3]
d_vec = [1e-3,2e-3,4e-3,6e-3]
#g_vec = [4,9,16]
g_vec = [3,6,9]
L_int_vec = [0.01,0.03]
u_rms_vec = [0.1,0.3,0.5]

cases = []
for d in d_vec:
    print(d)
    for g in g_vec:
        for L_int in L_int_vec:
            for u_rms in u_rms_vec:
                c = Case(d,g,L_int,u_rms)
                c.run()
                cases.append(c)
                
attrs = ['d','g','L_int','u_rms','beta','Fr','d_star','nondim_speed']
df = pd.DataFrame(index=range(len(cases)),columns=attrs)
for attr in attrs:
    df[attr] = [getattr(c,attr) for c in cases]
df['v_q'] = model.quiescent_speed(df['d'],df['g'],1)

fig,ax = plt.subplots()
ax.scatter(df['Fr']**-1*df['d_star']**-0.5*0.5,df['nondim_speed'],c=df['d_star'])
ax.plot([0,1],[0,1],ls='--',color='k',lw=1)

fig,ax = plt.subplots()
ax.scatter(df['Fr']*df['d_star']**0.5,df['nondim_speed'],c=df['d_star'])
ax.set_xscale('log')
ax.set_yscale('log')
x = np.linspace(0.5,3,51)
y = 0.5/x
ax.plot(x,y,ls='--',color='k',label=r'$\sim 1/(\mathrm{Fr} \sqrt{d^*})$')
ax.set_xlabel('$\mathrm{Fr} \sqrt{d^*}$')
ax.set_ylabel(r'$\langle v_z \rangle / v_\mathrm{q}$')
ax.legend()
fig.tight_layout()

stophere


# paper conditons
d_star = 0.12
L_int = 0.01
d = d_star*L_int

g = 10
Fr_vec = np.linspace(0.1,15,19)
cases = []
for Fr in Fr_vec:
    u_rms = Fr*np.sqrt(g*d)
    c = Case(d,g,L_int,u_rms)
    c.run()
    cases.append(c)
    
fig,ax = plt.subplots()
ax.plot(Fr_vec,[c.nondim_speed for c in cases],'o')
x = np.geomspace(2,15,301)
ax.plot(x,1.5/x)
ax.set_xlabel('$\mathrm{Fr} \sqrt{d^*}$')
ax.set_ylabel(r'$\langle v_z \rangle / v_\mathrm{q}$')
    
    
stopjere

vf = make_vf(n_modes=8,u_rms=0.2,L_int=0.01)

# image of the field
x = np.linspace(0,0.6,401)
y = np.linspace(0,1001,801)
X,Y = np.meshgrid(x,y)
Z = np.zeros_like(X)
xyz_flat = np.array([X.flatten(),Y.flatten(),Z.flatten()]).T
vel_flat = vf.get_velocity(3,xyz_flat)
vel = np.reshape(vel_flat,(np.shape(X)[0],np.shape(X)[1],3))
fig,axs = plt.subplots(1,3,figsize=(12,4))
for i,ax in enumerate(axs):
    ax.pcolormesh(X,Y,vel[:,:,i],cmap='seismic')
    ax.set_aspect('equal')
# import scipy.signal
# fig,ax = plt.subplots()
# for nper in [50,100,200,400,800,1200]:
#     f,pxx = scipy.signal.welch(vel[:,:,0],axis=-1,nperseg=nper,fs=1./(x[1]-x[0]))
#     ax.loglog(f,np.mean(pxx,axis=0),'x-')
# ax.axvline(1./(x[1]-x[0]))
# ax.axvline(1./L_int)
# ax.set_xlabel(r'$1/\Delta r$')

# x = np.linspace(0,10,10001)
# y = np.sin(x*10*(2*np.pi))
# f,pxx = scipy.signal.welch(y,fs=1./(x[1]-x[0]),nperseg=1000)
# fig,ax = plt.subplots()
# ax.loglog(f,pxx)
# plt.figure(); plt.plot(x,y)

# integral length scale
autocorrs = []
x = np.linspace(0,0.6,801)
y = np.linspace(0,1001,101)
X,Y = np.meshgrid(x,y)
Z = np.zeros_like(X)
xyz_flat = np.array([X.flatten(),Y.flatten(),Z.flatten()]).T
for t in np.random.uniform(0,10,100):
    #vf = make_vf(n_modes=8,u_rms=0.2,L_int=0.01)
    xyz_flat = np.array([X.flatten(),Y.flatten(),Z.flatten()]).T
    vel_flat = vf.get_velocity(t,xyz_flat)
    vel = np.reshape(vel_flat,(np.shape(X)[0],np.shape(X)[1],3))
    autocorr = (vel[:,:,0].T * vel[:,0,0]).T
    autocorr = np.mean(autocorr,axis=0)
    autocorrs.append(autocorr)
autocorrs = np.mean(np.array(autocorrs),axis=0)
autocorr = autocorrs/autocorrs[0]

plt.figure()
plt.plot(x,autocorr)

plt.figure()
plt.plot(x,np.cumsum(autocorr*np.gradient(x)))
    
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