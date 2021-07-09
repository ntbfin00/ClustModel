This code was written for the project: Discrete Dynamical Jeans Modelling of Simulated Galaxy Clusters (SUMMER 2021)

TNGextract
==========
For use with IllustrisTNG simulation TNG300-1 at z=0.

Extract all required ClustModel inputs (positions, masses, velocities, dispersions) from IllustrisTNG.

#### INPUT:
- Group number of selected cluster to model

#### OUTPUT:
- Projected x and y galaxy tracer positions (arcsecs)
- Spherical polar velocities of galaxy tracers (km/s)
- Line of sight velocities and dispersions (km/s)
- Stellar masses of galaxy tracers (Msun)
- Approximate concentration parameter 'c' of cluster
- M200 and R200 for the cluster (Msun and pc)
- 2 anisotropy parameters: interior to and outside of cluster scale radius
---------------------------------------------------------

ClustModel
============
Calculate the line of sight velocity and disperision for galaxy tracers in a cluster.

#### INPUT: Provided by TNGextract.py for Illustris clusters
- Projected x and y galaxy tracer positions
- 3D-positions of galaxy tracers
- Spherical polar velocities of galaxy tracers (not required for dispersion modelling)
- Line of sight velocities and dispersions of cluster (for comparison)
- Stellar mass of galaxy tracers

#### FITTING PARAMETERS: Provided by TNGextract.py for Illustris clusters
- Concentration parameter 'c' of NFW profile 
- M200 of NFW dark matter halo
- 2 anisotropy parameters: interior to and outside of cluster scale radius

#### OUTPUT:
- Modelled line of sight velocities and dispersions (km/s)
--------------------------------------------------------

ModelTEST
===========
Vary the 4 fitting parameters to test how well they are constrained.

#### INPUT: Provided by TNGextract.py for Illustris clusters
- 3D-positions of galaxy tracers
- Spherical polar velocities of galaxy tracers (not required for dispersion modelling)
- Stellar mass of galaxy tracers

#### OUTPUT:
- Plot of parameter variation with true Illustris parameter values marked
