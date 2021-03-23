# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 10:53:31 2021

@author: druth
"""

import pointparticlesinaflow as pb
from pointparticlesinaflow import analysis, equations
from pointparticlesinaflow.velocity_fields import gaussian
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import toolkit.parallel
import time

folder = r'E:\210204_new_pointbubble_data\gaussian_inertial_vs_visc\\'
figfolder = r'E:\210105_breakup_figures\\'

# create the velocity field


dstar = 0.12
#Fr_vec = np.concatenate([np.geomspace(0.1,6,10,endpoint=False),np.linspace(6,10,4)])
Fr_vec = np.array([0.1,0.25,0.5,0.75,1,1.5,2,3,4,5,6,7,8,9,10,11,12])
n_sim_vec = [15]*3 + [25]*3 + [40]*5 + [50]*6

res = []

mean_speeds = []

inertial = {}
visc =  {}
start_ix = 400 # t=2 given dt = 0.005

def load_inertial_and_visc(Fr,i):
    res = []
    mr_inertial = equations.MaxeyRileyPointBubbleConstantCoefs()
    for name_start,mr in zip(['gaussian_inertial_Fr',],[mr_inertial]):
        d,g = analysis.dg_given_nondim(Fr, dstar, vf.u_char, vf.L_char)
        sim_inertial = pb.Simulation(vf,{},{},mr)
        sim_inertial.add_data(folder+name_start+'{:03.4f}'.format(Fr)+'_dstar'+'{:03.4f}'.format(Fr)+'_v'+str(i)+'.pkl')
        a = analysis.CompleteSim(sim_inertial)
        res.append(a)
    return res

for Fr,n_sim in zip(Fr_vec,n_sim_vec):
        
    vf = gaussian.RandomGaussianVelocityField(n_modes=64,u_rms=1,L_int=1)
    print(Fr)
    
    mean_speeds_Fr = []
    
    def load_case(i):
        
        complete_sims = load_inertial_and_visc(Fr,i)
        
        res = []
        for a in complete_sims:        
            a_res = dict(v_q=a.v_q,
                         grav_z=a.grav_z,
                         d=a.d,
                         g=a.g,
                         n_sim=n_sim,
                         
                         mean_vz=np.mean(a['v'][start_ix:,:,2]),
                         std_vz=np.std(a['v'][start_ix:,:,2]),
                         mean_uz=np.mean(a['u'][start_ix:,:,2]),
                         std_uz=np.std(a['u'][start_ix:,:,2]),
                         mean_slip=np.mean(np.linalg.norm(a['v'][start_ix:,...]-a['u'][start_ix:,...],axis=-1)),
                         std_slip=np.std(np.linalg.norm(a['v'][start_ix:,...]-a['u'][start_ix:,...],axis=-1)),
                         mean_slipz=np.mean(a['v'][start_ix:,:,2]-a['u'][start_ix:,:,2]),
                         std_slipz=np.std(a['v'][start_ix:,:,2]-a['u'][start_ix:,:,2]),
                         
                         mean_dragz=np.mean(a['drag'][start_ix:,:,2]),
                         std_dragz=np.std(a['drag'][start_ix:,:,2]),
                         mean_pressz=np.mean(a['press'][start_ix:,:,2]),
                         std_pressz=np.std(a['press'][start_ix:,:,2]),                         
                         )
            
            res.append(a_res)
        return res
        
    res_Fr = toolkit.parallel.parallelize_job(load_case,np.arange(n_sim))
    df_inertial_Fr = pd.DataFrame([rfr[0] for rfr in res_Fr])
    inertial[Fr] = df_inertial_Fr.mean(axis=0)
    
inertial = pd.DataFrame.from_dict(inertial,orient='index')
df = inertial


fig,ax = plt.subplots()

# velocities
ax.loglog(df.index,df['mean_vz']/df['v_q'],ls='-',color='k',label=r'$\langle v_z \rangle$') # marker='o',
ax.loglog(df.index,df['mean_slip']/df['v_q'],ls='--',color='cyan',label=r'$\langle |\vec{v}-\vec{u}| \rangle$') # marker='+',
ax.loglog(df.index,df['mean_slipz']/df['v_q'],ls=':',color='darkred',label=r'$\langle v_z - u_z \rangle$') # marker='+',
#ax.loglog(df.index,df['mean_uz']/df['v_q'],ls=':',color='darkblue') # marker='o',

x = np.geomspace(3,13,51)
y = 2.6*x**-1
ax.plot(x,y,color='k',ls='-.')#,label='$\propto 1/\mathrm{Fr}$')
ax.text(4,0.70,'$\propto 1/\mathrm{Fr}$')
y = x*2./3
ax.plot(x,y,color='k',ls='-.')
ax.text(4,3.9,'$\propto \mathrm{Fr}$')


ax.legend(frameon=False)
ax.set_xlabel(r'''$\mathrm{Fr} = u'/\sqrt{gd}$''')
ax.set_ylabel('mean speed $/v_\mathrm{q}$')
fig.tight_layout()
fig.savefig(figfolder+'inertial_gaussian_figure.pdf')

#axs[0,0].loglog(df.index,df['mean_uz']/df['v_q'],marker='+',ls=ls,color='darkblue')

stophere

# plot 1/Fr scaling
x = np.geomspace(3,13,51)
y = 2*x**-1
axs[0,0].plot(x,y,color='r',ls=':',label='$\propto 1/\mathrm{Fr}$')

axs[0,0].legend(frameon=False)

axs[0,1].plot(np.nan,np.nan,'-',lw=5,color='orange',label=r'$\vec{F}_{\mathrm{drag},z}$')
axs[0,1].plot(np.nan,np.nan,'-',lw=5,color='purple',label=r'$\vec{F}_{\mathrm{press},z}$')
axs[0,1].legend(frameon=False)

axs[0,0].set_ylabel('mean speed $/v_\mathrm{q}$')
axs[1,0].set_ylabel('std. speed $/v_\mathrm{q}$')
axs[0,1].set_ylabel('mean force $/F_\mathrm{b}$')
axs[1,1].set_ylabel('std. speed $/F_\mathrm{b}$')
    
[ax.set_xlabel(r'''$\mathrm{Fr} = u'/\sqrt{gd}$''') for ax in axs.flatten()]
    
fig.tight_layout()
stophere
fig.savefig(figfolder+'inertial_vs_viscous_gaussian_speeds_forces.pdf')


'''
Single condition distribution of velocities
'''

Fr = 12
n_sim = 50
def get_case_arrays(i):
    complete_sims = load_inertial_and_visc(Fr,i)
    arrs = []
    for a in complete_sims:
        d = dict(uz=a['u'][start_ix:,:,2].flatten(),
                 vz=a['v'][start_ix:,:,2].flatten(),
                 dragz=a['drag'][start_ix:,:,2].flatten(),
                 pressz=a['press'][start_ix:,:,2].flatten(),
                 )
        arrs.append(d)
    return arrs

fig,axs = plt.subplots(2,2,figsize=(9,7))
Fr_vals = [0.5,10]
n_sims = [15,50]

for Fr,n_sim,axi in zip(Fr_vals,n_sims,[0,1]):

    
    complete_sims = load_inertial_and_visc(Fr,0)
    
    arrss = toolkit.parallel.parallelize_job(get_case_arrays,range(n_sim))
    arrs_inertial = [arrs[0] for arrs in arrss]
    arrs_visc = [arrs[1] for arrs in arrss]
    
    arrs_inertial = {key:np.concatenate([arrs[key] for arrs in arrs_inertial]) for key in arrs_inertial[0]}
    arrs_visc = {key:np.concatenate([arrs[key] for arrs in arrs_visc]) for key in arrs_visc[0]}

    # plot velocity distirubitons
    ax = axs[0,axi]
    for arrs,a,ls in zip([arrs_inertial,arrs_visc],complete_sims,['-','--']):
        
        # v_z
        x,y = analysis.get_hist(arrs['vz']/a.v_q,bins=101)
        ax.semilogy(x,y,color='k',ls=ls)
        
        # u_z
        x,y = analysis.get_hist(arrs['uz']/a.v_q,bins=101)
        ax.semilogy(x,y,color='b',ls=ls)
        
        # slip_z
        x,y = analysis.get_hist((arrs['vz']-arrs['uz'])/a.v_q,bins=101)
        ax.semilogy(x,y,color='g',ls=ls)
        
    ax.set_xlabel('speed $/ v_\mathrm{q}$ [-]')
    ax.set_ylabel('PDF [-]')
        
    # plot force distributions
    ax = axs[1,axi]
    for arrs,a,ls in zip([arrs_inertial,arrs_visc],complete_sims,['-','--']):
    
        # drag
        x,y = analysis.get_hist(arrs['dragz']/a.grav_z,bins=101)
        ax.semilogy(x,y,color='orange',ls=ls)
        
        # pressure
        x,y = analysis.get_hist(arrs['pressz']/a.grav_z,bins=101)
        ax.semilogy(x,y,color='purple',ls=ls)
        
        # sum
        x,y = analysis.get_hist((arrs['dragz']+arrs['pressz'])/a.grav_z,bins=101)
        ax.semilogy(x,y,color='gray',ls=ls)
    
    ax.set_xlabel('force $/ F_\mathrm{b}$ [-]')
    ax.set_ylabel('PDF [-]')
        
    axs[0,axi].set_title('$\mathrm{Fr}='+str(Fr)+'$')
    
#fig.tight_layout()
    
axs[0,0].plot(np.nan,np.nan,'-',lw=5,color='k',label=r'$v_z$')
axs[0,0].plot(np.nan,np.nan,'-',lw=5,color='b',label=r'$u_z$')
axs[0,0].plot(np.nan,np.nan,'-',lw=5,color='g',label=r'$v_z - u_z$')
axs[0,0].legend(frameon=False)

axs[1,0].plot(np.nan,np.nan,'-',lw=5,color='orange',label=r'$F_{\mathrm{drag},z}$')
axs[1,0].plot(np.nan,np.nan,'-',lw=5,color='purple',label=r'$F_{\mathrm{press},z}$')
axs[1,0].plot(np.nan,np.nan,'-',lw=5,color='gray',label=r'$F_{\mathrm{drag},z}+F_{\mathrm{press},z}$')
axs[1,0].legend(frameon=False)

fig.tight_layout()
fig.savefig(figfolder+'inertial_vs_viscous_gaussian_speeds_forces_distributions.pdf')