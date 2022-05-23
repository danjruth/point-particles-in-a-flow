# -*- coding: utf-8 -*-
"""
Created on Mon May 23 11:00:30 2022

@author: danjr
"""
import numpy as np
import pointparticlesinaflow as ppart
from pointparticlesinaflow.velocity_fields import jhtdb
from pointparticlesinaflow import equations, EquationOfMotion
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import scipy.optimize

vf = jhtdb.JHTDBVelocityField()

d_star_vals = np.geomspace(0.001,10,51)
#d_star_vals = np.array([0.01,])
d_vals = d_star_vals * vf.L_int
nu = vf.nu

fig,axs = plt.subplots(1,4,figsize=(15,4),sharex=True,sharey=False)

Fr_L_vals = [0.1,0.5,1,2]
colors = ['r','g','b','cyan']

for ax in axs:
    ax.axhline(1,color='k',lw=1)
    ax.axvline(1,color='k',lw=1)

for Fr_L,color in zip(Fr_L_vals,colors):
    
    g = (vf.u_rms/Fr_L)**2 / vf.L_int
    
    v_q_vals = np.zeros_like(d_vals)
    Re_q_vals = np.zeros_like(d_vals)
    
    for di,d in enumerate(d_vals):
            
        v_q = equations.calc_vq_Snyder(d,nu,g)
        v_q_vals[di] = v_q
        Re_q_vals[di] = v_q * d / nu
        
    beta_vals = vf.u_rms / v_q_vals
    Fr_vals = vf.u_rms / np.sqrt(d_vals * g)
    beta_eta_vals = (jhtdb.ISOTROPIC1024COARSE_PARAMS['eta'] / jhtdb.ISOTROPIC1024COARSE_PARAMS['T_eta']) / v_q_vals
       
    lab = Fr_L
    axs[0].loglog(d_star_vals,Re_q_vals,'-',label=lab,color=color)
    axs[1].loglog(d_star_vals,beta_vals,'-',label=lab,color=color)
    axs[2].loglog(d_star_vals,beta_eta_vals,'-',label=lab,color=color)
    axs[3].loglog(d_star_vals,Fr_vals,'-',label=lab,color=color)
    
    #ax.legend()
    
axs[0].legend(title = r'''$ u' / \sqrt{g L_\mathrm{int}}$''')

[ax.set_xlabel(r'$d / L_\mathrm{int}$') for ax in axs]
[ax.axvline(jhtdb.ISOTROPIC1024COARSE_PARAMS['eta'] / vf.L_int,color='gray',label=r'$\eta / L_\mathrm{int}$',ls='--',zorder=-np.inf) for ax in axs]

axs[1].set_ylabel(r'''$\beta = u'/v_\mathrm{q}$''')
axs[1].set_title(r'integral scale velocity to quiescent')
axs[2].set_ylabel(r'''$\beta_\eta = u_\eta/v_\mathrm{q}$''')
axs[2].set_title(r'Kolmogorov scale velocity to quiescent')
axs[3].set_ylabel(r'''$\mathrm{Fr} = u'/\sqrt{d g}$''')
axs[3].set_title(r'bubble-scale Froude number')

axs[0].set_ylabel(r'''$\mathrm{Re}_\mathrm{q} = v_\mathrm{q} d / \nu$''')
axs[0].set_title(r'quiescent Reynolds number')


                  
fig.tight_layout()

