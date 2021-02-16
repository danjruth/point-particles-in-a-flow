# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 09:53:48 2021

@author: druth
"""

import numpy as np
import matplotlib.pyplot as plt
from point_bubble_JHTDB import classes, analysis, equations
from point_bubble_JHTDB.velocity_fields import jhtdb

'''
Load bubble data
'''

vf = jhtdb.JHTDBVelocityField()
vf.init_field()

mr = equations.MaxeyRileyPointBubbleConstantCoefs()
              
# create the simulation
phys_params = {}
sim_params = {}
sim = classes.Simulation(vf,phys_params,sim_params,mr)
sim.add_data(r'E:\210204_new_pointbubble_data\Fr1_dstar0.12.pkl',include_velfield=False)
a = analysis.CompleteSim(sim,rotated=True)

'''
Load lagrangian data
'''

vf_L = jhtdb.JHTDBVelocityField()
vf_L.init_field()

eom = equations.LagrangianEOM()
              
# create the simulation
phys_params = {'d':np.nan,'g':np.nan}
sim_params = {}
sim_L = classes.Simulation(vf_L,phys_params,sim_params,eom)
sim_L.add_data(r'E:\210204_new_pointbubble_data\lagrangian.pkl',include_velfield=False)
a_L = analysis.CompleteSim(sim_L,rotated=True)
a_L.v_q = np.nan

a_L.grav_z = np.nan
a_L.v_q = np.nan

stophere

fig,ax = plt.subplots()
ax.plot(a['t'],a['v'][...,2].mean(axis=1)/a.v_q,color='k')
ax.plot(a_L['t'],a_L['v'][...,2].mean(axis=1)/a.v_q,color='gray')
ax.plot(sim.t,sim.v[...,2].mean(axis=1)/a.v_q,color='k',ls='--')
ax.plot(sim_L.t,sim_L.v[...,2].mean(axis=1)/a.v_q,color='gray',ls='--')

# fig,ax = plt.subplots()
# ax.plot(a['t']/a.T_vf,a['drag'][...,2].mean(axis=1)/a.grav_z,)
# ax.plot(a['t']/a.T_vf,a['press'][...,2].mean(axis=1)/a.grav_z,)
# ax.plot(a['t']/a.T_vf,a['lift'][...,2].mean(axis=1)/a.grav_z,)

# fig,ax = plt.subplots()
# ax.plot(a['t']/a.T_vf,a['drag'][...,2].std(axis=1)/a.grav_z,)
# ax.plot(a['t']/a.T_vf,a['press'][...,2].std(axis=1)/a.grav_z,)
# ax.plot(a['t']/a.T_vf,a['lift'][...,2].std(axis=1)/a.grav_z,)

# fig,ax = plt.subplots()
# i = 5
# ax.plot(a['t']/a.T_vf,a['drag'][:,i,2]/a.grav_z,)
# ax.plot(a['t']/a.T_vf,a['press'][:,i,2]/a.grav_z,)
# ax.plot(a['t']/a.T_vf,a['lift'][:,i,2]/a.grav_z,)

# velocity gradients
We_q = 1.
sigma_by_rho = a.d * a.v_q / We_q

def compute_We(velgrad,d,sigma_by_rho):
    axial_velgrad = np.moveaxis(np.array([velgrad[...,i,i] for i in range(3)]),0,-1)
    norm_val = np.linalg.norm(axial_velgrad,axis=-1)
    We = norm_val**2 * d**3 / sigma_by_rho
    return We

We_bub = compute_We(sim.velgrad,sim.d,sigma_by_rho)[1000:]
We_f = compute_We(sim_L.velgrad,sim.d,sigma_by_rho)[1:]

We_slip_bub = a.d * np.linalg.norm(a['v']-a['u'],axis=-1)**2 / sigma_by_rho
We_slip_f = a.d * np.linalg.norm(a_L['v']-a_L['u'],axis=-1)**2 / sigma_by_rho
We_slip_bub = We_slip_bub[1000:4800]
We_slip_f = We_slip_f[1:4800]


fig,axs = plt.subplots(1,3,figsize=(12,4))
for We,We_slip,ls,lab in zip([We_bub,We_f],[We_slip_bub,We_slip_f],['-','--'],['bubbles','fluid tracers']):
    
    x,y  = analysis.get_hist(We.flatten(),bins=np.geomspace(1e-2,300,51))
    axs[0].loglog(x,y,color='r',ls=ls,label=lab)
    x,y  = analysis.get_hist(We_slip.flatten(),bins=np.geomspace(1e-2,300,51))
    axs[0].loglog(x,y,color='b',ls=ls,label=lab)

    We_crit_values = np.geomspace(0.1,400,17)
    
    all_t_until_above = []
    times = np.arange(len(We))*sim.dt
    for wi,We_crit in enumerate(We_crit_values):
        print(We_crit)
        t_until_above = []
        for i in np.arange(1000):
            #print(i)
            above_crit = We[:,i]>We_crit
            t_above_crit = times[above_crit]
            
            for ti in range(len(times)):
                t = times[ti]
                t_above_after = t_above_crit[t_above_crit>=t]
                if len(t_above_after)>0:
                    t_until_above.append(np.min(t_above_after-t))
                else:
                    t_until_above.append(np.nan)
        
        t_until_above = np.array(t_until_above)
        all_t_until_above.append(t_until_above)
        
        # show distirbution of times
        if wi in [0,6,11]:
            x,y = analysis.get_hist(t_until_above[~np.isnan(t_until_above)],bins=np.geomspace(2e-3,3,51),cumulative=True)
            color = [0.5,0.5,wi/(len(We_crit_values)-1)]
            if ls=='-':
                label='{:03.3f}'.format(We_crit)
            else:
                label=None
            axs[1].semilogx(x[1:]/vf.T_char,y[1:],ls=ls,color=color,label=label)
        
        
    axs[2].loglog(We_crit_values,np.nanmean(all_t_until_above,axis=1)/vf.T_char,'-x',color='k',ls=ls)
    
axs[0].set_title('distribution of $\mathrm{We}$')
axs[0].set_xlabel(r'$\mathrm{We}$')
axs[0].set_ylabel('cumulative PDF [-]')
axs[0].legend(frameon=False)
    
axs[1].set_title('chance of experiencing given $\mathrm{We}_\mathrm{c}$ over duration')
axs[1].set_xlabel(r'$\Delta t / T_\mathrm{int}$')
axs[1].set_ylabel('cumulative PDF [-]')
axs[1].legend(title='$\mathrm{We}_\mathrm{c}$',frameon=False)

axs[2].set_title('mean time until experiencing given $\mathrm{We}_\mathrm{C}$')
axs[2].set_xlabel(r'$\mathrm{We}_\mathrm{c}$')
axs[2].set_ylabel(r'$\langle \Delta t / T_\mathrm{int} \rangle$')

fig.tight_layout()
fig.savefig(r'E:\210105_breakup_figures\\bubble_vs_tracer_We_dstar0.1_Fr1.pdf')