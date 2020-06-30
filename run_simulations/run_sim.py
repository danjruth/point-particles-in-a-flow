import numpy as np
import point_bubble_JHTDB as pb
from point_bubble_JHTDB.model import PointBubbleSimulation

params = {'beta':0.5,
         'A':0.1,
         'Cm':0.5,
         'Cd':0.5,
         'Cl':0.25,
         'n_bubs':500,
         'dt_factor':0.5,}

m = PointBubbleSimulation(params)

m.init_sim()

m.run_model()