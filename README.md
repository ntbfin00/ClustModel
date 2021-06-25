ClustModel
============
This code calculates the line of sight velocity and disperision for galaxy tracers in a cluster.

#### INPUT:
- Projected positions of galaxy tracers (in arcsecs)
- M200 and R200 of cluster DM halo  (in Msun and pc)
- Anisotropy parameters for cluster

#### OUTPUT:
- Line of sight velocities and dispersions (km/s)
--------------------------------------------------------
TNGextract
==========
For use with IllustrisTNG simulation TNG300-1 at z=0.

Extract all required ClustModel inputs (positions, masses, velocities, dispersions) from IllustrisTNG.

#### OUTPUT: 
- 'positions.dat' - projected x and y galaxy tracer positions (in arcsecs)
