# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 09:53:48 2021

@author: druth
"""

import numpy as np
import matplotlib.pyplot as plt
from point_bubble_JHTDB import classes, analysis, equations
from point_bubble_JHTDB.velocity_fields import jhtdb
import toolkit.parallel
import pandas as pd

folder = r'E:\210204_new_pointbubble_data\HIT_viscous_varyingdstar\\'
figfolder = r'E:\210105_breakup_figures\\'
case_name = 'visc_dstar0.03.pkl'

dstar_vals = [0.01,0.03,0.05,0.07,0.10]
start_ix = 1000

vf = jhtdb.JHTDBVelocityField()
vf.init_field()

mr = equations.MaxeyRileyPointBubbleConstantCoefsVisc()

complete_sims = []
for dstar in dstar_vals:
    print(dstar)
    sim = classes.Simulation(vf,{},{},mr)
    sim.add_data(folder+'visc_dstar'+'{:03.2f}'.format(dstar)+'.pkl',include_velfield=True)
    a = analysis.CompleteSim(sim,rotated=True)
    complete_sims.append(a)

'''
Load bubble data
'''
df = pd.DataFrame(index=np.arange(len(complete_sims)))

fig,axs = plt.subplots(1,3,figsize=(11,5))
for ai,a in enumerate(complete_sims):
    a.dstar = a.d/a.L_vf
    a.beta = a.u_vf / a.v_q
    
    u_norm = a.u_vf
    
    lab = r'$d^* = '+'{:03.2f}'.format(a.dstar)+r'$ ($\beta='+'{:03.2f}'.format(a.beta)+'$)'
    
    print(a.dstar,a.beta)
    
    # mean vertical velocity
    axs[0].plot(a['t']/a.T_vf,a['v'][:,:,2].mean(axis=1)/u_norm,label=lab)
    
    # mean slip velocity
    l=axs[1].plot(a['t']/a.T_vf,(a['v'][:,:,2]-a['u'][:,:,2]).mean(axis=1)/u_norm,label='{:03.2f}'.format(a.dstar))[0]
    mean = (a['v'][:,:,2]-a['u'][:,:,2]).mean(axis=1)/u_norm
    std = (a['v'][:,:,2]-a['u'][:,:,2]).std(axis=1)/u_norm
    #axs[1].fill_between(a['t']/a.T_vf,mean-std,mean+std,label='{:03.2f}'.format(a.dstar),color=l.get_color(),alpha=0.3)
    
    # mean sampled velocity
    axs[2].plot(a['t']/a.T_vf,a['u'][:,:,2].mean(axis=1)/u_norm,label='{:03.2f}'.format(a.dstar))
    
    df.loc[ai,'std_vz'] = a['v'][start_ix:,:,2].std()
    df.loc[ai,'std_uz'] = a['u'][start_ix:,:,2].std()
    df.loc[ai,'std_slipz'] = (a['v'][start_ix:,:,2]-a['u'][start_ix:,:,2]).std()
    df.loc[ai,'u_prime'] = a.u_vf
    df.loc[ai,'v_q'] = a.v_q
    df.loc[ai,'beta'] = a.beta
    df.loc[ai,'dstar'] = a.dstar
    
axs[0].legend()

[ax.set_xlabel('$t/T_\mathrm{int}$') for ax in axs]
axs[0].set_ylabel(r'$\langle v_z \rangle / v_\mathrm{q}$')
axs[1].set_ylabel(r'$\langle v_z - u_z \rangle / v_\mathrm{q}$')
axs[2].set_ylabel(r'$\langle u_z \rangle / v_\mathrm{q}$')

axs[0].set_title('mean vertical velocity')
axs[1].set_title('mean vertical slip velocity')
axs[2].set_title('mean sampled vertical fluid velocity')

[ax.set_ylim(-0.55,1.25) for ax in axs]

fig.tight_layout()
#fig.savefig(figfolder+r'HIT_viscous_meanverticalvels_vs_time.pdf')

fig,ax = plt.subplots()

x = np.geomspace(1e-3,1e-2,51)
ax.plot(x,x,ls='--',color='k')
# create the simulation


start_ix = 3000
fig,ax = plt.subplots()
mean_pressure = [a['press'][start_ix:,:,2].mean()/a.grav_z for a in complete_sims]
mean_drag = [a['drag'][start_ix:,:,2].mean()/a.grav_z for a in complete_sims]
beta = [a.beta for a in complete_sims]
dstar = [a.dstar for a in complete_sims]
ax.plot(dstar,mean_pressure,'-x',color='purple')
ax.plot(dstar,mean_drag,'-o',color='orange')