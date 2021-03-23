# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 15:35:34 2021

@author: druth
"""

import numpy as np
from ..classes import VelocityField

class InterpolatedSteadyVelocity(VelocityField):
    '''Time-invariant velocity field defined by values on a grid
    '''
    
    def __init__(self,interp_dicts,name='interpolated'):
        self.interp_dicts = interp_dicts
        
    def get_velocity(t,xyz):
        pass