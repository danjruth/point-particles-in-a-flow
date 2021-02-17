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

# create the velocity field


dstar = 0.12
#Fr_vec = np.concatenate([np.geomspace(0.1,6,10,endpoint=False),np.linspace(6,10,4)])
Fr_vec = np.array([0.1,0.25,0.5,0.75,1,1.5,2,3,4,5,6,7,8,9,10,11,12])
n_sim_vec = [15]*3 + [25]*3 + [40]*5 + [50]*6
#Fr_vec = np.array([2])

res = []

for Fr,n_sim in zip(Fr_vec,n_sim_vec):
    
    def do_job(i):
    
        vf = gaussian.RandomGaussianVelocityField(n_modes=64,u_rms=1,L_int=1)
        vf.init_field()
        
        '''
        Inertial drag
        '''
        d,g = analysis.dg_given_nondim(Fr, dstar, vf.u_char, vf.L_char)
        phys_params = {'d':d,
                        'g':g,
                        'Cm':0.5,
                        'Cd':1,
                        'Cl':0.0}
        sim_params = {'n_bubs':150,
                      'dt':5e-3,
                      't_min':0,
                      't_max':15,
                      'fname':'inertial'}
        mr = equations.MaxeyRileyPointBubbleConstantCoefs()
        sim_inertial = pb.Simulation(vf,phys_params,sim_params,mr)
        sim_inertial.init_sim(g_dir='random',pos_lims=((0,0,0),(1000,1000,1000)),)
        sim_inertial.run()
        sim_inertial.save(folder+'gaussian_inertial_Fr'+'{:03.4f}'.format(Fr)+'_dstar'+'{:03.4f}'.format(Fr)+'_v'+str(i)+'.pkl')
        #del sim_inertial
        #a = analysis.CompleteSim(sim_inertial)
        
        '''
        Viscous drag
        '''
        phys_params = {'d':sim_inertial.d,
                        'g':sim_inertial.g,
                        'Cm':0.5,
                        'Cd':np.nan,
                        'Cl':0.0,
                        'nu':analysis.nu_given_quiescent_visc(sim_inertial.d,sim_inertial.g,sim_inertial.v_q)}
        mrv = equations.MaxeyRileyPointBubbleConstantCoefsVisc()
        sim_visc = pb.Simulation(vf,phys_params,sim_params,mrv)
        sim_visc.init_sim()
        sim_visc.g_dir = sim_inertial.g_dir
        sim_visc.x[0,:,:] = sim_inertial.x[0,:,:]
        sim_visc.run()
        #b = analysis.CompleteSim(sim_visc)
        sim_visc.save(folder+'gaussian_visc_Fr'+'{:03.4f}'.format(Fr)+'_dstar'+'{:03.4f}'.format(Fr)+'_v'+str(i)+'.pkl')
        
        del sim_inertial, sim_visc
        
        #del a.r['dudt'], a.r['grav']
        #del b.r['dudt'], b.r['grav']
        
        return None
    
    res_Fr = toolkit.parallel.parallelize_job(do_job,range(n_sim))
    #for i,x in enumerate(res_Fr):
    #    si,sv = x
    #    si.save(folder+'gaussian_inertial_Fr'+'{:03.4f}'.format(Fr)+'_dstar'+'{:03.4f}'.format(Fr)+'_v'+str(i)+'.pkl')
    #    sv.save(folder+'gaussian_viscous_Fr'+'{:03.4f}'.format(Fr)+'_dstar'+'{:03.4f}'.format(Fr)+'_v'+str(i)+'.pkl')
    #res.append(res_Fr)
    
    
res_inertial = []
res_visc = []
for fri,Fr in enumerate(Fr_vec):
    res_Fr = res[fri]
    res_inertial.append([res_Fr[i][0] for i in range(len(res_Fr))])
    res_visc.append([res_Fr[i][1] for i in range(len(res_Fr))])
del res

ix_start = 4000
    
mean_speed_inertial = []
mean_speed_visc = []
std_speed_inertial = []
std_speed_visc = []
for fri,Fr in enumerate(Fr_vec):
    
    mean_speed_inertial.append( np.nanmean([res_inertial[fri][i]['v'][ix_start:,:,2].mean()/res_inertial[fri][i].v_q for i in range(len(res_inertial[fri]))]) )
    mean_speed_visc.append( np.nanmean([res_visc[fri][i]['v'][ix_start:,:,2].mean()/res_inertial[fri][i].v_q for i in range(len(res_visc[fri]))]) )
    
    std_speed_inertial.append( np.nanstd([res_inertial[fri][i]['v'][ix_start:,:,2].mean()/res_inertial[fri][i].v_q for i in range(len(res_inertial[fri]))]) )
    std_speed_visc.append( np.nanstd([res_visc[fri][i]['v'][ix_start:,:,2].mean()/res_inertial[fri][i].v_q for i in range(len(res_inertial[fri]))]) )

mean_speed_inertial = np.array(mean_speed_inertial)
mean_speed_visc = np.array(mean_speed_visc)
std_speed_inertial = np.array(std_speed_inertial)
std_speed_visc = np.array(std_speed_visc)

fig,ax = plt.subplots()

ax.plot(Fr_vec,mean_speed_inertial,'x-',label='non-linear drag, $C_\mathrm{D}=1$',color='r')
ax.fill_between(Fr_vec,mean_speed_inertial-std_speed_inertial,mean_speed_inertial+std_speed_inertial,color='r',alpha=0.5)

ax.plot(Fr_vec,mean_speed_visc,'o--',label='linear drag, $C_\mathrm{D}=24/\mathrm{Re}$',color='b')
ax.fill_between(Fr_vec,mean_speed_visc-std_speed_visc,mean_speed_visc+std_speed_visc,color='b',alpha=0.5)


ax.set_xlabel(r'''$\mathrm{Fr} = u'/\sqrt{gd}$''')
ax.set_ylabel(r'$\langle v_z \rangle / v_\mathrm{q}$')
ax.legend()
fig.tight_layout()

fig,ax = plt.subplots()
fri = -1
ax.plot(res_inertial[fri][0]['t'],res_inertial[fri][0]['x'][:,0,:]-res_inertial[fri][0]['x'][0,0,:],ls='-')
ax.plot(res_visc[fri][0]['t'],res_visc[fri][0]['x'][:,0,:]-res_visc[fri][0]['x'][0,0,:],ls='--')

fig,ax = plt.subplots()
ax.plot(res_visc[fri][0]['x'][:,0,0],res_visc[fri][0]['x'][:,0,2],color='b')
ax.plot(res_inertial[fri][0]['x'][:,0,0],res_inertial[fri][0]['x'][:,0,2],color='r')
ax.set_aspect('equal')