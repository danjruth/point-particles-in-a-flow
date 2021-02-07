# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 11:49:32 2021

@author: druth
"""

from point_bubble_JHTDB import model, analysis, data
from point_bubble_JHTDB.velocity_fields import gaussian
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import toolkit.parallel

'''
Construct the velocity field
'''
vf = gaussian.RandomGaussianVelocityField(n_modes=12,u_rms=1,L_int=1)
u_rms = vf.calc_u_rms(n_t=100,componentwise=True)

print(u_rms)
print((np.std(vf.b)+np.std(vf.c))/2)

print(vf.get_velocity_gradient(0,np.atleast_2d([3,4,5])))
print(vf.get_velocity_gradient_numerical(0,np.atleast_2d([3,4,5])))

print(vf.get_dudt(0,np.atleast_2d([3,4,5])))
print(vf.get_dudt_numerical(0,np.atleast_2d([3,4,5])))

stophere
# vf.save(data.data_dir+'test_vf.pkl')
# print(vf.get_velocity(0,np.atleast_2d([4,6,2])))

# vf2 = gaussian.load_random_gaussian_velocity_field(data.data_dir+'test_vf.pkl')
# print(vf2.get_velocity(0,np.atleast_2d([4,6,2])))

# x = np.linspace(0,12,121)
# y = np.linspace(0,10,101)
# X,Y = np.meshgrid(x,y)
# Z = np.zeros_like(X)

# stds = []
# for _ in range(20):
#     vf = gaussian.RandomGaussianVelocityField(n_modes=12,u_rms=1,L_int=1)
#     vel = vf.get_2d_velocity_field(np.random.uniform(0,20),X,Y,Z)
#     print(vel.std())
#     stds.append(vel.std())
# print(np.mean(stds))

# spatial autocorrelation - single velocity field

n_fields = 10
fig,axs = plt.subplots(1,2,)
for _ in np.arange(n_fields):
    n_autocorrs = 50
    autocorrs = []
    vf = gaussian.RandomGaussianVelocityField(n_modes=24,u_rms=1,L_int=1)
    x = np.linspace(0,100,1001)
    y = np.linspace(0,100,81)
    X,Y = np.meshgrid(x,y)
    Z = np.zeros_like(X)
    for _ in np.arange(n_autocorrs):
        velfield = vf.get_2d_velocity_field(np.random.uniform(0,10000), X+np.random.uniform(0,10000), Y, Z)
        autocorr = (velfield[:,0,0]*velfield[:,:,0].T).T
        autocorrs.append(np.mean(autocorr,axis=0))
    autocorrs = np.array(autocorrs)
    autocorr = np.mean(autocorrs,axis=0)
    autocorr = autocorr/autocorr[0]
    axs[0].plot(x,autocorr,label='one velocity field')
    axs[1].plot(x,np.cumsum(autocorr*np.gradient(x)))

n_fields = 10
autocorrss = []
for _ in np.arange(n_fields):
    vf = gaussian.RandomGaussianVelocityField(n_modes=24,u_rms=1,L_int=1)
    autocorrs = []
    for _ in range(n_autocorrs):
        velfield = vf.get_2d_velocity_field(np.random.uniform(0,10000), X+np.random.uniform(0,10000), Y, Z)
        autocorr = (velfield[:,0,0]*velfield[:,:,0].T).T
        autocorrs.append(np.mean(autocorr,axis=0))
    autocorrs = np.array(autocorrs)
    autocorr = np.mean(autocorrs,axis=0)
    autocorr = autocorr/autocorr[0]
    autocorrss.append(autocorr)
autocorrss = np.array(autocorrss)
autocorr_all = np.mean(autocorrss,axis=0)
axs[0].plot(x,autocorr_all,label='10 velocity fields')
axs[1].plot(x,np.cumsum(autocorr_all*np.gradient(x)))

[ax.set_xlabel(r'$\Delta x / L$') for ax in axs]
axs[0].set_ylabel(r'autocorrelation')
axs[1].set_ylabel(r'integral of autocorrelation')
fig.tight_layout()


'''
parameterization of slip velocity
'''
vf = gaussian.RandomGaussianVelocityField(n_modes=24,u_rms=1,L_int=1)


#stophere

mr = model.MaxeyRileyPointBubbleConstantCoefs()
#mr = model.LagrangianEOM()

#Fr_vec = np.concatenate([np.geomspace(0.01,2,5,endpoint=False),np.linspace(2,10,5)])
Fr_vec = np.concatenate([np.geomspace(0.05,7,12),np.arange(4,12,1)])
n_fieldss =[18]*len(Fr_vec)
css = []
for Fr,n_fields in zip(Fr_vec,n_fieldss):
    
    cs = []
    
    def do_job(_):
        vf = gaussian.RandomGaussianVelocityField(n_modes=12,u_rms=1,L_int=1)
        dstar = 0.1
        d,g = model.bubble_params_given_nondim(Fr, dstar, 1, 1)
        bubble_params = {'d':d,
                        'g':g,
                        'Cm':0.5,
                        'Cd':1,
                        'Cl':0.0}
        
        sim_params = {'n_bubs':300,
                      'dt':1e-3,
                      't_min':0,
                      't_max':4,
                      'fname':'test'}
        
        sim = model.Simulation(vf,bubble_params,sim_params,mr)
        sim.init_sim()
        sim.run()
        a = analysis.CompleteSim(sim,norm=False)
        slip_fluc =  (a['v'][1000:]-a['v'][1000:].mean(axis=(0,1))) - (a['u'][1000:]-a['u'][1000:].mean(axis=(0,1)))
        slip_fluc_norm = np.linalg.norm(slip_fluc,axis=-1)
        mean_sfn = np.mean(slip_fluc_norm)
        plt.figure()
        plt.plot(a['t'],a['v'][:,0,:]/a.v_q)
        a.mean_sfn = mean_sfn
        
        key_del = ['vort','dudt','u_times_deldotu','press','drag','lift','slip']
        for key in key_del:
            del a.r[key]
        return a
    
    cs = toolkit.parallel.parallelize_job(do_job,range(n_fields))
    css.append(cs)
        
Frs = [cs[0].Fr for cs in css]
nondim_speeds = [np.mean([c['v'][1000:,:,2].mean()/c.v_q for c in cs]) for cs in css]
y_val = [np.mean([c['v'][1000:,:,2].mean() * c.mean_sfn / c.v_q**2 for c in cs]) for cs in css]

fig,ax = plt.subplots()
ax.semilogy(Frs,y_val,'x-')
ax.set_xlabel(r'$\mathrm{Fr}$')
ax.set_ylabel(r'''$\langle v_z \rangle \langle |\vec{v}' - \vec{u}' | \rangle / v_\mathrm{q}^2 $''')

nondim_sfns = [np.mean([c.mean_sfn / c.u_vf for c in cs]) for cs in css]
fig,ax = plt.subplots()
ax.plot(Frs,nondim_sfns,'-x')
ax.set_xlabel('$\mathrm{Fr}$')
ax.set_ylabel(r'''$\langle |\vec{v}' - \vec{u}' | \rangle / u' $''')

plt.figure()
plt.loglog(Frs,nondim_speeds,'x-')

'''
Variable drag coefficient
'''
vf = gaussian.RandomGaussianVelocityField(n_modes=24,u_rms=1,L_int=1)
print(vf.calc_mean_flow())
dstar = 0.1
Fr = 0.5

d,g = model.bubble_params_given_nondim(Fr, dstar, 1, 1)
bubble_params = {'d':d,
                'g':g,
                'Cm':0.5,
                'Cd':1,
                'Cl':0.0}

sim_params = {'n_bubs':600,
              'dt':1e-3,
              't_min':0,
              't_max':20,
              'fname':'test'}

# simulation with naive Cd = 1
sim_const = model.Simulation(vf,bubble_params,sim_params,model.MaxeyRileyPointBubbleConstantCoefs())
sim_const.init_sim(g_dir='z')
sim_const.run()

# simulation with variable Cd, where Cd=1 is achieved for the quiescent case
sim_var = model.Simulation(vf,bubble_params,sim_params,model.MaxeyRileyPointBubbleVariableCoefs())
sim_var.nu = model.nu_given_Req(sim_const.d,sim_const.g,sim_const.Cd,121.2)
sim_var.init_sim(g_dir='z')
sim_var.run()

# calculate mean Cd from variable Cd case
c_const = analysis.CompleteSim(sim_const)
c_var = analysis.CompleteSim(sim_var)
slip = c_var['v']-c_var['u']
slip_norm = np.linalg.norm(slip,axis=-1)
Re = slip_norm*c_var.d/sim_var.nu
Cd = model.calc_Cd_Snyder(Re.flatten())
mean_Cd = np.mean(Cd[~np.isinf(Cd)])
fig,ax = plt.subplots()
ax.hist(Cd[~np.isinf(Cd)],density=True,bins=501)
ax.axvline(mean_Cd,color='r')
ax.axvline(sim_const.Cd,color='k')

# simulation with a constant Cd equal to the mean Cd from the variable Cd simulation
bubble_params['Cd'] = mean_Cd
sim_const_newCd = model.Simulation(vf,bubble_params,sim_params,model.MaxeyRileyPointBubbleConstantCoefs())
sim_const_newCd.init_sim(g_dir='z')
sim_const_newCd.run()
c_const_newCd = analysis.CompleteSim(sim_const_newCd)

# compare, normalizing by the quiescent rise velocity in Cd_quiescent case
for c in [c_const,c_var,c_const_newCd]:
    print(c.g,c.d,c.Cd,c.v_q,c.beta)
    print(c['v'].mean(axis=(0,1))/c_const.v_q)

    
stophere

def make_vf(n_modes=8,u_rms=0.1,L_int=0.01):
    return gaussian.RandomGaussianVelocityField(n_modes=n_modes,u_rms=u_rms,L_int=L_int)

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
            sim.nondim_speed = nondim_speed
            return sim
        
        self.sims = np.array(toolkit.parallel.parallelize_job(do_job, range(n)))
        self.nondim_speed = np.mean([j.nondim_speed for j in self.sims])
        
#c = Case(2e-3,9.81,0.01,0.1)
#c.run()
#stophere


vf = make_vf(n_modes=12,u_rms=0.1,L_int=0.01)
#sim = 
x = np.linspace(0,.3,91)
y = np.linspace(0,0.4,101)
X,Y = np.meshgrid(x,y)
Z = np.zeros_like(X)
v2d = vf.get_2d_field(3, X, Y, Z)
plt.figure()
plt.pcolormesh(X,Y,v2d[:,:,0],cmap='seismic')
stpohere
#vf.get_velocity(3,np.atleast_2d([3,2,1]))
#stophere
sim = model.Simulation(vf,bubble_params,sim_params,mr)
sim.init_sim(vz_0='v_q')
sim.run()    
a = analysis.CompleteSim(sim,norm=True)
plt.figure();plt.plot(a['t'],a['v'][:,:,:].mean(axis=1),ls='-')
stophere

vf = make_vf()

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