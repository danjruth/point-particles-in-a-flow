# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 12:34:47 2020

@author: ldeike
"""

import numpy as np
from point_bubble_JHTDB.model import *

d_by_L = lam_by_Lint
beta = 0.5
Cd = 0.5

# calculate "physical" parameters
d_1 = d_by_L * L_int
v_q_1 = u_rms/beta
g_1 = (3./4) * Cd * v_q_1**2/d_1

# calculate A
A = u_rms**2 / (g_1*L_int)


# what's done in init_sim

g = (u_rms**2/L_int)/A
v_q = u_rms/beta
d = (3./4) * Cd * v_q**2 / g