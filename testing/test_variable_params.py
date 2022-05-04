# -*- coding: utf-8 -*-
"""
Created on Mon May  2 15:18:49 2022

@author: danjr
"""

import numpy as np
import matplotlib.pyplot as plt

from pointparticlesinaflow.velocity_fields import gaussian
import pointparticlesinaflow as ppart
from pointparticlesinaflow import equations, EquationOfMotion

import toolkit.parallel

import timeit



class BasicEOM(EquationOfMotion):
    
    additional_pkeys = ['Cm']

    def __init__(self,):
        super().__init__(name='basic_eom',
                         forces=[#equations.DragForceSnyder2007(),
                                 equations.ConstantCDDragForce(),
                                 #equations.ConstantCLLiftForce(),
                                 equations.PressureForceBubble(),
                                 equations.GravForceBubble()])
        
    def calc_m_eff(self,r):
        return r['Cm']*(r['d']/2)**3*4./3*np.pi
    
class Run:
    
    def __init__(self,n_modes,u_rms,L_int,n_sims=30):
        
        self.n_sims = n_sims
        
        self.n_modes = n_modes
        self.u_rms = u_rms
        self.L_int = L_int
        
        self.T_int = L_int/u_rms

        self.eom = BasicEOM()
        
        
    def init_sim(self,):
        
        sims = []
        
        for _ in range(self.n_sims):
            n = 20
            #self.min_d = 1e-4
            #self.max_d = 3e-3
            #d = np.random.uniform(min_d,max_d,n)
            #logd = np.random.uniform(np.log10(self.min_d),np.log10(self.max_d),n)
            #d = 10**logd
            d = np.ones(n)*3e-4
            Cd = 0.5 * np.ones_like(d)
            Cm = np.ones_like(d)
            Cl = 0.5 * np.ones_like(d)
            nu = np.ones_like(d) * 8.9e-7
            g = 9.81
            g_dir = np.zeros((n,3))
            g_dir[:,-1] = -1
            
            part_params = dict(d=d,Cd=Cd,Cm=Cm,Cl=Cl,nu=nu,g=g,g_dir=g_dir)
            dt = np.min([0.001,self.T_int/100])
            sim_params = dict(t_min=0,t_max=np.max([0.4,4*self.T_int]),dt=dt,n=n)
            
            vf = gaussian.RandomGaussianVelocityField(n_modes=self.n_modes,u_rms=self.u_rms,L_int=self.L_int)
            vf.init_field()
            
            sim = ppart.Simulation(vf,self.eom,part_params,sim_params)
            sim.init_sim()
            sim.x[0,:,:] = np.random.uniform(0,self.L_int*100,size=(n,3))
            
            sims.append(sim)
        
        self.sims = sims
        
    def run(self):
        
        def run_i(a):
            a.run(disp=False)
            return a
        #self.sim.run(disp=True)
        self.sims = toolkit.parallel.parallelize_job(run_i,self.sims,threads=10,max_nbytes=1e9)
        
    def get_mean_vz(self):
        mean_vz_vals = []
        for i in range(self.n_sims):
            sim = self.sims[i]
            t_cond = (sim.t>0.1) & (sim.t > self.T_int*2)
            mean_vz_vals.append(sim.v[t_cond,:,2].mean())
        self.mean_vz = np.mean(mean_vz_vals)
        return self.mean_vz

    
    def get_mean_vz_curve(self,bins=9):
        
        mean_vzs = []
        d_bins = np.geomspace(self.min_d,self.max_d,bins)
        self.d_centers = d_bins[:-1] + np.diff(d_bins)/2
        for i in range(self.n_sims):
            sim = self.sims[i]
            vz_mean = np.zeros_like(self.d_centers)
            d = sim.p['d']
            t_cond = (sim.t>0.1) & (sim.t > self.T_int*2)
            for di in range(len(d_bins)-1):
                cond = (d>=d_bins[di]) & (d<d_bins[di+1])
                vz_mean[di] = sim.v[t_cond][:,cond,2].mean()
            mean_vzs.append(vz_mean)
        self.vz_mean = np.mean(mean_vzs,axis=0)
        return self.d_centers,self.vz_mean
    
    def get_std_vz_curve(self,bins=9):
        
        std_vzs = []
        d_bins = np.geomspace(self.min_d,self.max_d,bins)
        self.d_centers = d_bins[:-1] + np.diff(d_bins)/2
        for i in range(self.n_sims):
            sim = self.sims[i]
            vz_std = np.zeros_like(self.d_centers)
            d = sim.p['d']
            t_cond = (sim.t>0.1) & (sim.t > self.T_int*2)
            for di in range(len(d_bins)-1):
                cond = (d>=d_bins[di]) & (d<d_bins[di+1])
                vz_std[di] = sim.v[t_cond][:,cond,2].std()
            std_vzs.append(vz_std)
        self.vz_std = np.mean(std_vzs,axis=0)
        return self.d_centers,self.vz_std
    
runs = [Run(1,0.0001,0.0001),
        Run(12,0.1,0.0001),
        Run(12,0.1,0.0003),
        Run(12,0.1,0.0005),
        Run(12,0.1,0.001),
        Run(12,0.1,0.002),
        Run(12,0.1,0.003),
        Run(12,0.1,0.005),
        Run(12,0.1,0.01),
        Run(12,0.1,0.05),
]

n_cases = len(runs)-1
colors = ['k'] + [[0.5,0.5,i/(n_cases-1)] for i in range(0,n_cases)]

for a in runs:
    
    a.init_sim()
    a.run()
    a.get_mean_vz()
    
mean_vz_nondim = [a.mean_vz/runs[0].mean_vz for a in runs[1:]]
Lint_vals = [a.L_int for a in runs[1:]]

plt.figure()
plt.semilogx(Lint_vals,mean_vz_nondim,'-o')
    
stophere

fig,ax = plt.subplots()

for a,color in zip(runs,colors):
    x,y = a.get_mean_vz_curve()
    lab = r'$n_\mathrm{modes}='+str(int(a.n_modes))+'$, $u_\mathrm{rms} = '+'{:1.2f}'.format(a.u_rms)+'$, $L_\mathrm{int} = '+'{:1.3f}'.format(a.L_int)+'$'
    ax.plot(x,y,'-o',label=lab,color=color)
    #stophere
    
ax.legend()
    
stophere

fig,ax = plt.subplots()

def color_d(d):
    return np.array([0.5,0.5,(d-min_d)/(max_d-min_d)])

for i in range(90):
    ax.plot(sim.t,sim.x[:,i,0],color=color_d(d[i]),alpha=0.7)
    
slip = sim.v - sim.u
slip_mag = np.linalg.norm(slip,axis=-1)
Re = sim.p['d'] * slip_mag / sim.p['nu']
plt.figure(); plt.scatter(sim.p['d'],Re[-1,:],alpha=0.5)

d_bins = np.geomspace(1e-4,1e-2,15)
d_centers = d_bins[:-1] + np.diff(d_bins)/2
vz_mean = np.zeros_like(d_centers)
for di in range(len(d_bins)-1):
    cond = (d>=d_bins[di]) & (d<d_bins[di+1])
    vz_mean[di] = sim.v[-1,cond,2].mean()
plt.figure()
plt.plot(d_centers,vz_mean,'-o')


# '''
# Test speed
# '''

# n = 500

# slip = np.random.uniform(0,1,size=(n,3))
# slip_correct = np.moveaxis(slip,0,-1)
# slip_mag = np.linalg.norm(slip,axis=1)
# Cd = np.random.uniform(n)
# d = np.random.uniform(n)
# def compute():
#     drag = -1/8 * Cd * np.pi * d**2 * (np.moveaxis(slip,0,-1)*slip_mag)        
#     drag = np.moveaxis(drag,0,-1)
#     return drag

# def compute_better():
#     drag = -1/8 * Cd * np.pi * d**2 * (slip_correct*slip_mag)        
#     return drag

# n_compute = 30000
# t1 = timeit.timeit(compute,number=n_compute)
# print(t1/n_compute)
# t2 = timeit.timeit(compute_better,number=n_compute)
# print(t2/n_compute)
# print(t2/t1)