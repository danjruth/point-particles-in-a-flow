# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 16:55:07 2021

@author: druth
"""



import pointparticlesinaflow as pb
#stophere

from pointparticlesinaflow import analysis, equations
from pointparticlesinaflow.velocity_fields import gaussian, two_dimensional
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import toolkit.parallel
import time
import stereo_bubbles.velocity_analysis as va

d_exp_vec = np.array([1e-3,2e-3,4e-3])

Fr_vec = np.geomspace(0.5,13,10)

all_res = []
for d_exp in d_exp_vec:
    
    for Fr in Fr_vec:
        
        def do_job(i):
    
            vf = gaussian.RandomGaussianVelocityField(n_modes=64,u_rms=1,L_int=1)
            vf.init_field()
            
            # d and g given Fr and dstar
            d,g = analysis.dg_given_nondim(Fr, 0.1, vf.u_char, vf.L_char)
            
            # nu: match d^3 g / nu^2
            nu_sim = np.sqrt( d**2 * g / (d_exp**3 * va.g / va.nu_w**2) )
            
            # common simulation parameters
            sim_params = {'n_bubs':2,
                          'dt':1e-3,
                          't_min':0,
                          't_max':14,
                          'fname':'none'}
            
            # '''
            # Snyder, quiescent field
            # '''
            phys_params = {'d':d,
                            'g':g,
                            'Cm':0.5,
                            'nu':nu_sim, #analysis.nu_given_Req(d, g, 0.7, 1000), # pick nu given the Reynolds number that corresponds to a given velocity
                            'Cl':0.0}
            eom = equations.MaxeyRileyPointBubbleSnyder2007()
            vf_q = pb.VelocityField()
            sim_params_q = sim_params.copy()
            sim_params_q['n_bubs'] = 1
            simq = pb.Simulation(vf_q,phys_params,sim_params_q,eom)
            simq.init_sim(g_dir='random')
            #simq.v[0] = np.array([0.001,0.001,0.001])
            t1 = time.time()
            simq.run()
            print(time.time()-t1)
            a_q = analysis.CompleteSim(simq,rotated=True)
            a_q['v'].mean(axis=(0,1))
            v_q = np.mean(a_q['v'][-1,:,2])
            Re_q = d*v_q/nu_sim
            Cd_q = equations.calc_Cd_Snyder(Re_q)[0][0]
            
            
            #stophere
            '''
            Snyder, random gaussian field
            '''
            sim_params['n_bubs'] = 150
            eom = equations.MaxeyRileyPointBubbleSnyder2007()
            sim2 = pb.Simulation(vf,phys_params,sim_params,eom)
            sim2.init_sim(g_dir='random',pos_lims=((0,0,0),(1000,1000,1000)))
            t1 = time.time()
            sim2.run()
            print(time.time()-t1)
            a2 = analysis.CompleteSim(sim2,rotated=True)
            print('Snyder: ',str(a2['v'].mean(axis=(0,1))/v_q))
            
            '''
            Constant Cd, matching mean Cd from Snyder
            '''
            
            Re = np.linalg.norm((a2['v']-a2['u']),axis=-1) * a2.d / a2.nu
            Cd = equations.calc_Cd_Snyder(Re)
            Cd_mean = Cd.mean()
            fig,ax = plt.subplots()
            ax.hist(Cd.flatten(),bins=np.geomspace(0.01,10,101),cumulative=False,density=True,alpha=0.5,color='k')
            ax.axvline(Cd_q,color='k',ls='--')
            ax.axvline(Cd_mean,color='k',ls='-')
            ax.hist(Re.flatten(),bins=np.geomspace(0.1,10000,101),cumulative=False,density=True,alpha=0.5,color='r')
            ax.axvline(Re_q,color='r',ls='--')
            ax.axvline(Re.mean(),color='r',ls='-')
            ax.set_xscale('log')
            ax.set_yscale('log')
            
            phys_params = {'d':sim2.d,
                            'g':sim2.g,
                            'Cm':sim2.Cm,
                            'Cd':Cd_mean,
                            'Cl':sim2.Cl}
            eom = equations.MaxeyRileyPointBubbleConstantCoefs()
            sim3 = pb.Simulation(vf,phys_params,sim_params,eom)
            sim3.init_sim()
            sim3.g_dir = sim2.g_dir.copy()
            sim3.x[0] = sim2.x[0].copy()
            t1 = time.time()
            sim3.run()
            print(time.time()-t1)
            a3 = analysis.CompleteSim(sim3,rotated=True)
            print('Matching mean Cd from Snyder: ',str(a3['v'].mean(axis=(0,1))/v_q))
            
            plt.figure()
            plt.plot(a2['t'],a2['v'][:,0,:])
            plt.plot(a3['t'],a3['v'][:,0,:],ls='--')
            
            # res = {'a_q':a_q,
            #        'a_snyder':a2,
            #        'a_const':a3,
            #        'd_exp':d_exp,
            #        'Fr':Fr}
            
            res = {'v_q':a_q['v'][-1,:,2].mean(),
                   'd_exp':d_exp,
                   'Fr':Fr,
                   'mean_vz_snyder':a2['v'][2000:,:,2].mean(),
                   'mean_vz_const':a3['v'][2000:,:,2].mean(),
                   'Re_q':Re_q,
                   'Cd_q':Cd_q,
                   'Cd_mean':Cd_mean}
            
            return res
           
        
        print(d_exp,Fr)
        cond_res = toolkit.parallel.parallelize_job(do_job, np.arange(34))
        all_res.append(cond_res)

        # Gaussian velocity field
        
fig,axs = plt.subplots(1,2,figsize=(9,4))

colors = ['r','g','b']
for d_exp,color in zip(d_exp_vec,colors):
    res_use = [res for res in all_res if res[0]['d_exp']==d_exp]
    
    vznorm_snyder = []
    vznorm_const = []
    Cd_q = []
    Cd_mean = []
    for Fr in Fr_vec:
        res = [res for res in res_use if res[0]['Fr']==Fr][0]
        #v_q = res[0]['a_q']['v'][-1,:,2].mean()
        v_q = res[0]['v_q']
        #all_vz_norm_snyder = [np.mean(r['a_snyder']['v'][1000:,:,2]/v_q) for r in res]
        all_vz_norm_snyder = [np.mean(r['mean_vz_snyder']/v_q) for r in res]
        vznorm_snyder.append(np.mean(all_vz_norm_snyder))
        #all_vz_norm_const = [np.mean(r['a_const']['v'][1000:,:,2]/v_q) for r in res]
        all_vz_norm_const = [np.mean(r['mean_vz_const']/v_q) for r in res]
        vznorm_const.append(np.mean(all_vz_norm_const))
        Cd_q.append(np.mean([np.mean(r['Cd_q']) for r in res]))
        Cd_mean.append(np.mean([np.mean(r['Cd_mean']) for r in res]))
        
    axs[0].plot(Fr_vec,vznorm_snyder,'o-',color=color)
    axs[0].plot(Fr_vec,vznorm_const,'x--',color=color)
    
    #axs[1].plot(Fr_vec,Cd_q,'+:',color=color)
    axs[1].axhline(Cd_q[0],ls=':',color=color)
    axs[1].plot(Fr_vec,Cd_mean,'x--',color=color)
    
[ax.set_xlabel(r'''$\mathrm{Fr}=u'/\sqrt{dg}$''') for ax in axs]
axs[0].set_ylabel(r'$\langle v_z \rangle / v_\mathrm{q}$')
axs[1].set_ylabel(r'$\langle C_\mathrm{D} \rangle$')

axs[0].plot(np.nan,np.nan,'-o',color='k',label=r'$C_\mathrm{D} = C_\mathrm{D,Snyder}$')
axs[0].plot(np.nan,np.nan,'x--',color='k',label=r'$C_\mathrm{D} = \langle C_\mathrm{D,Snyder} \rangle$')

for d_exp,color in zip(d_exp_vec,colors):
    axs[0].plot(np.nan,np.nan,color=color,lw=5,label=r'$d = $ '+str(int(d_exp*1000))+' mm')

axs[1].plot(np.nan,np.nan,':',color='k',label='quiescent, Snyder')
axs[1].plot(np.nan,np.nan,'x--',color='k',label='turbulent, Snyder')

axs[0].legend(frameon=False)
axs[1].legend(frameon=False)

fig.tight_layout()
fig.savefig(va.figfolder+'gaussian_snyderCdVsConst.pdf')