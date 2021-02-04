# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 16:50:22 2020

@author: danjr
"""

import numpy as np
from ..model import VelocityField

try:
    from pyJHTDB import libJHTDB
    lJHTDB = libJHTDB()
    lJHTDB.initialize(exit_on_error=True)
    lJHTDB_available = True
except ImportError:
    print('Unable to import pyJHTDB and initialize')
    lJHTDB_available = False

# only define the velocity field class for this if the interface to the JHTDB
# is available
if lJHTDB_available:
    
    class JHTDBVelocityField(VelocityField):
        
        def __init__(self,data_set='isotropic1024coarse'):
            #VelocityField.__init__(self,name='JHTDB_'+data_set)
            super().__init__(name='JHTDB_'+data_set)
            self.data_set = data_set
            self.lJHTDB = lJHTDB
            
            # store physical and simulation propertie
            if data_set=='isotropic1024coarse':
                self.u_rms = 0.686
                self.L_int = 1.364
                self.T_int = self.L_int/self.u_rms
                self.eta = 0.00280
                self.T_eta = 0.0424
                self.lam = 0.113
                
                self.u_char = self.u_rms
                self.L_char = self.L_int
                self.T_char = self.T_int
                
                self.dx = 2*np.pi / 1024.
                self.dt = 0.002 # the timestep at which the DNS data is stored, = 10*dt_orig
                self.dt_orig = 0.0002
                self.t_min = 0.
                self.t_max = 10.
            
        def get_velocity(self,t,x,lJHTDB=None):
            if lJHTDB is None:
                lJHTDB = self.lJHTDB
            return lJHTDB.getData(t, point_coords=x.copy().astype(np.float32), data_set=self.data_set, getFunction='getVelocity', sinterp='Lag4', tinterp='PCHIPInt')

        def get_velocity_gradient(self,t,x,lJHTDB=None):
            
            if lJHTDB is None:
                lJHTDB = self.lJHTDB
        
            # query the data
            result_grad = lJHTDB.getData(t, x.copy().astype(np.float32), data_set=self.data_set,sinterp='FD4Lag4', tinterp='PCHIPInt',getFunction='getVelocityGradient')
            
            # put it into the correct shape
            result_grad_new = np.zeros((len(x),3,3)).astype(np.float32)
            for i in range(len(x)):
                this_velgrad = np.array([result_grad[i,0:3],result_grad[i,3:6],result_grad[i,6:]])
                result_grad_new[i,...] = this_velgrad
                
            return result_grad_new
        
        def get_dudt(self,t,x,u_t=None,delta_t=1e-4,lJHTDB=None):
            
            if lJHTDB is None:
                lJHTDB = self.lJHTDB
            
            if u_t is None:
                u_t = self.get_velocity(t,x,)
            u_tplusdeltat = self.get_velocity(t+delta_t,x,)
            dudt = (u_tplusdeltat - u_t) / delta_t
            return dudt
    
        
#     def get_velocity_filtered(t,x,filter_width,lJHTDB=lJHTDB):
#         #print(x.astype(np.float32))
#         return lJHTDB.getBoxFilter(t, point_coords=x.copy().astype(np.float32), data_set='isotropic1024coarse', field='velocity', filter_width=filter_width)
    
# #     def test_velocity_filtered():
# #         x = np.array([2,3.2,1.9])
# #         t = 4.2
# #         print('a')
# #         a = get_velocity_filtered(t,x.copy(),0.1)
# #         print(a)
# #         a = get_velocity_filtered(t,x.copy(),0.01)
# #         print(a)
# #         a = get_velocity(t,x.copy())
# #         print(a)
# #         return None
    
    
#     def get_velocity_gradient_filtered(t,x,filter_width):
        
#         # query the data
#         result_grad = lJHTDB.getBoxFilterGradient(t, x.copy().astype(np.float32), data_set='isotropic1024coarse', filter_width=filter_width, field='velocity',)
#         print(np.shape(result_grad))
        
#         # put it into the correct shape
#         result_grad_new = np.zeros((len(x),3,3)).astype(np.float32)
#         for i in range(len(x)):
#             this_velgrad = np.array([result_grad[i,0:3],result_grad[i,3:6],result_grad[i,6:]])
#             result_grad_new[i,...] = this_velgrad
            
#         return result_grad_new
    
#     def myVelocityGradient(t,point_coords,delta=1e-4,lJHTDB=lJHTDB):
#         '''
#         Compute the velocity gradient numerically. Check the effect of delta, it should be around 1e-4.
#         '''
#         points_plus = np.zeros((len(point_coords),3,2))
#         for i in np.arange(3):
#             points_plus[:,i,0] = point_coords[:,i]-delta
#             points_plus[:,i,1] = point_coords[:,i]+delta
        
#         points_flat = np.array([[points_plus[:,0,0],point_coords[:,1],point_coords[:,2]],
#                                 [points_plus[:,0,1],point_coords[:,1],point_coords[:,2]],
#                                 [point_coords[:,0],points_plus[:,1,0],point_coords[:,2]],
#                                 [point_coords[:,0],points_plus[:,1,1],point_coords[:,2]],
#                                 [point_coords[:,0],point_coords[:,1],points_plus[:,2,0]],
#                                 [point_coords[:,0],point_coords[:,1],points_plus[:,2,1]]])
        
#         points_flat = np.moveaxis(points_flat,-1,0)    
#         points_flat = np.reshape(points_flat, (len(point_coords)*6,3))
    
#         u_flat = get_velocity(t,points_flat,lJHTDB=lJHTDB)
#         u = np.reshape(u_flat,(len(point_coords),6,3))
        
#         vel_grad = np.zeros((len(point_coords),3,3)) # [point,component,grad_dir]    
#         for j in np.arange(len(point_coords)):
#             # for each point in time        
#             for i in np.arange(3):
#                 # for each velocity direction
#                 vel_grad[j,i,0] = (u[j,1,i]-u[j,0,i])/(2*delta)
#                 vel_grad[j,i,1] = (u[j,3,i]-u[j,2,i])/(2*delta)
#                 vel_grad[j,i,2] = (u[j,5,i]-u[j,4,i])/(2*delta)
                
#         return vel_grad
    
#     def get_sphere_kernel(r,n_per_r):
#         x = np.linspace(-r,r,n_per_r*2+1)
#         y = np.linspace(-r,r,n_per_r*2+1)
#         z = np.linspace(-r,r,n_per_r*2+1)

#         X,Y,Z = np.meshgrid(x,y,z)
#         XYZ = np.moveaxis(np.array([X,Y,Z]),0,-1)
#         XYZ_flat = np.reshape(XYZ,(np.shape(XYZ)[0]**3,3))

#         dist = np.linalg.norm(XYZ_flat,axis=1)
#         XYZ_use = XYZ_flat[dist<=r]
#         return XYZ_use
    
#     def get_points_in_sphere(center,sphere_kernel):
#         points = np.array([center[i]+sphere_kernel for i in range(len(center))])
#         return points
    
#     def get_velocity_volavg(t,x,sphere_kernel,lJHTDB=lJHTDB):
#         points = get_points_in_sphere(x,sphere_kernel)
#         #print(np.shape(points))
#         vels = get_velocity(t,points,lJHTDB=lJHTDB) # [bubble,point_in_sphere,axis]
#         #print(np.shape(vels))
#         return np.mean(vels,axis=1)
    
#     def get_velocity_gradient_volavg(t,x,sphere_kernel,lJHTDB=lJHTDB):
#         points = get_points_in_sphere(x,sphere_kernel)
#         #print('shape of points:')
#         #print(np.shape(points))
#         velgrads = np.array([get_velocity_gradient(t,points[i],lJHTDB=lJHTDB) for i in np.arange(len(x))]) # [bubble,point_in_sphere,axis]
#         #print('shape of velgrads:')
#         #print(np.shape(velgrads))
#         return np.mean(velgrads,axis=1) 
    
#     def surface_average_of_velocity(t,x,d,filter_size,d2_factor=1.01,avg1=None,lJHTDB=lJHTDB):
        
#         r = d/2
#         r1 = r
#         r2 = r1*d2_factor
#         #r2 = r1 + dx*2
        
#         if avg1 is None:
#             avg1 = get_velocity_filtered(t,x,filter_size,lJHTDB=lJHTDB)
#         avg2 = get_velocity_filtered(t,x,filter_size*d2_factor,lJHTDB=lJHTDB)
        
#         larger = r2**3 * avg2
#         smaller = r1**3 * avg1
#         deriv = (larger-smaller)/(r2-r1)
#         surface_avg = 1./(3*r1**2) * deriv
        
#         return surface_avg