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
