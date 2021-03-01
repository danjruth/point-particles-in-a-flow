# point-particles-in-a-flow

Simulate the motion of point-like particles in a flow.

## Overview

The main class is the `Simulation`, which encapsulates one simulation of any number of point-like particles in a single flow. The `Simulation` relies on a `VelocityField` and an `EquationOfMotion`, all of which are defined in `model.py`.


### Velocity Fields

Derived classes of `VelocityField` include
* `JHTDBVelocityField`, which employs the [`pyJHTDB`](https://github.com/idies/pyJHTDB) package to access the direct numerical simulations of turbulence available from the Johns Hopkins Turbulence Database
* `RandomGaussianVelocityField`, which constructs a field composed of a number of space- and time-dependent Fourier modes, with amplitudes, wavenumbers, and frequencies picked from random distributions


### Equations of motion
The `EquationOfMotion` is used to update the particles' velocities, given their current velocities and the flow conditions they experience. Derived classes of `EquationOfMotion` include
* `LagrangianEOM`, which is used to obtain Lagrangian fluid parcel trajectories by simply setting the particle velocity `v` equal to the fluid velocity `u` at the particle location
* `MaxeyRileyPointBubbleConstantCoefs`, which applies the Maxey-Riley equation simplified for a bubble in a much denser liquid, with constant lift, drag, and added-mass coefficients

## Example usage

### Create a Gaussian velocity field and simulation using the Maxey-Riley equation for a point bubble

```python
from pointparticlesinaflow import model, analysis
from pointparticlesinaflow.velocity_fields import gaussian
import matplotlib.pyplot as plt

# create the velocity field
vf = gaussian.RandomGaussianVelocityField(n_modes=12,u_rms=1,L_int=1)
vf.init_field()

# create the equation of motion
mr = model.MaxeyRileyPointBubbleConstantCoefs()

# define parameters for the bubbles simulated
bubble_params = {'d':0.1,
                'g':2,
                'Cm':0.5,
                'Cd':1,
                'Cl':0.0}
sim_params = {'n_bubs':20,
              'dt':1e-3,
              't_min':0,
              't_max':2,
              'fname':'example_simulation'}
              
# create the simulation
sim = model.Simulation(vf,bubble_params,sim_params,mr)

# initialize it (involves choosing the 20 bubbles' initial positions and defining each's gravity direction)
sim.init_sim()
```

### Run the simulation and analyze the results

```python
# simulate the bubbles between the t_min and t_max specified in sim_params
sim.run()

# save the data to a file
sim.save('example_simulation.pkl')

# create a CompleteSim object to analyze the results
# this rotates all vectors so the final entry of the last axis is parallel to gravity
csim = analysis.CompleteSim(sim)

# plot the mean vertical velocity of the bubbles against time
# normalize time by the characteristic scale of the velocity field
# normalize velocities by the quiescent velocity of the bubble
fig,ax = plt.subplots()
ax.plot(csim['t']/csim.T_vf,np.mean(csim['v'][:,:,2],axis=1))
ax.set_xlabel(r'$t/T_\mathrm{velocityfield}$')
ax.set_ylabel(r'$\langle v_z \rangle / v_\mathrm{q}$')
```

### Reload the data later for analysis

```python
# initialize the object with a stand-in velocity field of the same type that is to be loaded
# the EOM must be specified again (it can't be saved easily), but the parameters aren't necessary
sim_reloaded = model.Simulation(gaussian.RandomGaussianVelocityField(),{},{},mr)

# add the data that was saved
sim_reloaded.add_data('example_simulation.pkl', include_velfield=True)

# sim_reloaded.run() can be called to restart the simulation if it wasn't complete upon saving
```
