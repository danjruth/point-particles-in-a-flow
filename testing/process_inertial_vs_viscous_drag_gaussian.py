# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 10:53:31 2021

@author: druth
"""

import point_bubble_JHTDB as pb
from point_bubble_JHTDB import analysis, equations
from point_bubble_JHTDB.velocity_fields import gaussian
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

for Fr,n_sim in zip(Fr_vec,n_sim_vec):
        
    vf = gaussian.RandomGaussianVelocityField(n_modes=64,u_rms=1,L_int=1)
    print(Fr)
    
    mean_speeds_Fr = []
    
    def load_case(i):
        
        res = {}
        mr_inertial = equations.MaxeyRileyPointBubbleConstantCoefs()
        mr_visc = equations.MaxeyRileyPointBubbleConstantCoefsVisc()
        res = []
        for name_start,mr in zip(['gaussian_inertial_Fr','gaussian_visc_Fr'],[mr_inertial,mr_visc]):
            d,g = analysis.dg_given_nondim(Fr, dstar, vf.u_char, vf.L_char)
            sim_inertial = pb.Simulation(vf,{},{},mr)
            sim_inertial.add_data(folder+name_start+'{:03.4f}'.format(Fr)+'_dstar'+'{:03.4f}'.format(Fr)+'_v'+str(i)+'.pkl')
            a = analysis.CompleteSim(sim_inertial)
        
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
    df_visc_Fr = pd.DataFrame([rfr[1] for rfr in res_Fr])
    inertial[Fr] = df_inertial_Fr.mean(axis=0)
    visc[Fr] = df_visc_Fr.mean(axis=0)
    
inertial = pd.DataFrame.from_dict(inertial,orient='index')
visc = pd.DataFrame.from_dict(visc,orient='index')

fig,axs = plt.subplots(2,2,figsize=(9,7))
for df,ls in zip([inertial,visc],['-','--']):
    
    # velocities
    axs[0,0].loglog(df.index,df['mean_vz']/df['v_q'],ls=ls,color='k') # marker='o',
    axs[0,0].loglog(df.index,df['mean_slip']/df['v_q'],ls=ls,color='cyan') # marker='+',
    #axs[0,0].loglog(df.index,df['mean_uz']/df['v_q'],marker='+',ls=ls,color='darkblue')
    
    # stds of velocity
    axs[1,0].loglog(df.index,df['std_vz']/df['v_q'],ls=ls,color='k') # marker='o',
    axs[1,0].loglog(df.index,df['std_slip']/df['v_q'],ls=ls,color='cyan') # marker='+',
    
    # forces
    axs[0,1].semilogx(df.index,df['mean_dragz']/df['grav_z'],color='orange',ls=ls) # marker='x',
    axs[0,1].semilogx(df.index,df['mean_pressz']/df['grav_z'],color='purple',ls=ls) # marker='^',
    
    # stds of forces
    axs[1,1].loglog(df.index,df['std_dragz']/df['grav_z'],color='orange',ls=ls) # marker='x',
    axs[1,1].loglog(df.index,df['std_pressz']/df['grav_z'],color='purple',ls=ls) # marker='^',

    
axs[0,0].plot(np.nan,np.nan,'-',color='gray',label='inertial (non-linear) drag')
axs[0,0].plot(np.nan,np.nan,'--',color='gray',label='viscous (linear) drag')
axs[0,0].plot(np.nan,np.nan,color='white',label=' ')
axs[0,0].plot(np.nan,np.nan,'-',lw=5,color='k',label=r'$v_z$')
axs[0,0].plot(np.nan,np.nan,'-',lw=5,color='cyan',label=r'$|\vec{v}-\vec{u}|$')
axs[0,0].plot(np.nan,np.nan,color='white',label=' ')


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
fig.savefig(figfolder+'inertial_vs_viscous_gaussian_speeds_forces.pdf')