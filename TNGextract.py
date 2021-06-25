#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 09:01:25 2021

@author: nathanfindlay
"""
import numpy as np
import matplotlib.pyplot as plt
import requests
from astropy.cosmology import FlatLambdaCDM
from astropy import table, units as u

# Define astronomical constants
G = 4.3009125e-3 # (km/s)^2 pc/Msun
c = 2.998e5  #km/s
redshift=0
scale_factor = 1.0 / (1+redshift)
h = 0.6774 # choose your H0 
H0 = h*100 # km/Mpc/s
cosmo = FlatLambdaCDM(H0=H0, Om0=0.2726)

# Conversions for working with Illustris data
to_pc=(1e3*scale_factor/h)  # from ckpc/h to pc
to_Msun=(1e10/h)  # from 10^10Msun/h to Msun
pc_to_km = 3.086e13  # from pc to km

baseUrl = 'http://www.tng-project.org/api/'
sim = 'TNG300-1/'  # use TNG300-1 simulation
snapNum=99  # snapshot number equivalent to z=0


#=================EDIT THIS SECTION WITH YOUR DETAILS===================
headers = {"api-key":"56fb3d0452d97e788e9fb75ef3bff17b"}#{"api-key":"YOUR_API_KEY"}  # insert your API key for IllustrisTNG

filepath='/Users/nathanfindlay/Gals_SummerProject/data_files/'#'/YOUR_FILEPATH'  # path where output files should be stored

zobs=0.1
Dist = (c*zobs/H0)*1e6  # set observer distance in pc (set here for cluster at z=0.1)

grnr = 1  # set FoF group number to work with (grnr=0 is not a relaxed cluster)
#=======================================================================


# Define 'get' function to make API requests
def get(path, params=None):
    # make HTTP GET request to path
    r = requests.get(path, params=params, headers=headers)

    # raise exception if response code is not HTTP SUCCESS (200)
    r.raise_for_status()

    if r.headers['content-type'] == 'application/json':
        return r.json() # parse json responses automatically

    if 'content-disposition' in r.headers:
        filename = filepath + r.headers['content-disposition'].split("filename=")[1]
        with open(filename, 'wb') as f:
            f.write(r.content)
        return filename # return the filename string

    return r


print('FoF group '+str(grnr))

# Extract tracer positions for chosen group
min_mass = 1e9/(1e10/h)  # choose subhalos that host galaxies by mass
url = baseUrl + sim + 'snapshots/99/subhalos/'
search_query = '?grnr=' + str(grnr) + '&mass_stars__gt=' + str(min_mass)

subhalos = get(url+search_query, {'limit':100000, 'order_by':'-mass'})  # order subhalos by mass
num = len(subhalos['results'])
print('Subhalos in group: ',subhalos['count'],';  Results returned: ',num)

xpos=np.zeros(num)
ypos=np.zeros(num)
zpos=np.zeros(num)

for i in range(0,num):
    my_subs = get(subhalos['results'][i]['url'])
    xpos[i] = my_subs['pos_x']*to_pc  # spatial positions within the periodic box in pc
    ypos[i] = my_subs['pos_y']*to_pc
    zpos[i] = my_subs['pos_z']*to_pc
    
# Extract position of primary subhalo
url = baseUrl + sim + 'snapshots/99/halos/'+str(grnr)+'/info.json'
fof_halo = get(url)
pri_indx = fof_halo['GroupFirstSub']  # index of primary/central subhalo
url = baseUrl + sim + 'snapshots/99/subhalos/'+str(pri_indx)
pri_sub = get(url)
xc = pri_sub['pos_x']*to_pc  # coordinates of centre in pc
yc = pri_sub['pos_y']*to_pc
zc = pri_sub['pos_z']*to_pc

# Calculate tracer positions relative to centre
xrel = xpos-xc  
yrel = ypos-yc
zrel = zpos-zc


# Calculate projected tracer positions in arcsecs
xproj = np.arctan((xrel)/(Dist+(zrel)))
yproj = np.arctan((yrel)/(Dist+(zrel)))
print('Projected positions determined')


#===============================================================
# Determine radial distance from centre to tracer.
Rdist = np.sqrt((xrel)**2+(yrel)**2+(zrel)**2)  # in pc
Rmax = np.max(Rdist)  # distance of furthest tracer
#===============================================================



# Extract FoF halo properties
url = baseUrl + sim + 'snapshots/99/halos/'+str(grnr)+'/info.json'
fof_halos = get(url)

M200=fof_halos['Group_M_Mean200']*to_Msun  # in Msun
R200=fof_halos['Group_R_Mean200']*to_pc  # in pc
print('M200 and R200 extracted')

# Extract velocities
vx_obs=np.zeros(num)
vy_obs=np.zeros(num)
vz_obs=np.zeros(num)

vcx = pri_sub['vel_x']  # velocity of central subhalo in km/s
vcy = pri_sub['vel_y']
vcz = pri_sub['vel_z']

for i in range(0,num):
    my_subs = get(subhalos['results'][i]['url'])
    vx_obs[i] = my_subs['vel_x']-vcx  
    vy_obs[i] = my_subs['vel_y']-vcy
    vz_obs[i] = my_subs['vel_z']-vcz  # velocity along line of sight (LoS)
print('Velocities calculated')

# Bin velocities and calculate dispersions for each annulus.
nsig_bin = 20  # number of velocity bins for dispersion calculations (same should be used in model later)
sig_bin = np.linspace(0, Rmax, nsig_bin)  # make evenly distributed bins
bin_indx = np.digitize(Rdist, sig_bin, right=True)
sigma_obs=np.zeros(len(Rdist))

for i in range(0,nsig_bin+1):
    bin_objs = np.where(bin_indx == i)[0]  # determine what tracers lie in each annulus
    v_ms = np.mean(vz_obs[bin_objs]**2)  # mean of the square velocities
    v_sm = np.mean(vz_obs[bin_objs])**2  # square of the mean velocities
    sigma_obs[bin_objs] = np.sqrt(v_ms - v_sm)  # set sigma for each tracer in annulus
print('Dispersions calculated')

# Extract subhalo stellar masses
msub=np.zeros(num)

for i in range(0,num):
    my_subs = get(subhalos['results'][i]['url'])
    msub[i] = my_subs['mass_stars']*to_Msun
print('Subhalo stellar masses extracted')

# Write positions, LoS velocities, LoS dispersions and masses to file
units = 'x\' (arcsecs)   y\' (arcsecs)   vz (km/s)   sigma_z (km/s)   Mstellar (Msun)'
data = np.column_stack((xproj,yproj,vz_obs,sigma_obs,msub))  # in pc, no z as observing along z axis

np.savetxt(filepath + 'obs.dat', data, header=units)
print('Observations saved to file')

# Read in observations from file and add appropriate units
obs = table.QTable.read(filepath + "obs.dat", format="ascii", names=["x","y","vz","sigz","M"])  
obs["x"].unit = u.arcsec
obs["y"].unit = u.arcsec
obs["vz"].unit = u.km/u.s
obs["sigz"].unit = u.km/u.s
obs['M'].unit = u.solMass
  
# Plot velocity and dispersion maps
col_lim = 2500  # limit of colorbar

plt.figure(figsize=(5,8))

plt.subplot(2,1,1)
plt.title('observed $v_z \,(km\,s^{-1})$',fontsize=27)
plt.scatter(obs["x"],obs["y"],c=obs['vz'],cmap='jet',s=9)
plt.xlabel('x (arcsec)',fontsize=20)
plt.ylabel('y (arcsec)',fontsize=20)
plt.clim(-col_lim, col_lim)
cbar=plt.colorbar(orientation="vertical")

plt.subplot(2,1,2)
plt.title('observed $\sigma_z\, (km\,s^{-1})$',fontsize=27)
plt.scatter(obs["x"],obs["y"],c=obs['sigz'],cmap='jet',s=9)
plt.xlabel('x\' (arcsec)',fontsize=20)
plt.ylabel('y\' (arcsec)',fontsize=20)
plt.clim(0, col_lim*(3/4))
cbar=plt.colorbar(orientation="vertical")

plt.tight_layout()
plt.show()

# Determine real anisotropy parameters for observations
def v_sph(r,x,y,z,vx,vy,vz):  # convert velocities from cartesian to spherical
    r=r[1:]  # remove primary subhalo from calculation 
    x=x[1:]
    y=y[1:]
    z=z[1:]
    vx=vx[1:]
    vy=vy[1:]
    vz=vz[1:]
    
    vr = (x*vx+y*vy+z*vz)/r
    vphi = (x*vy-y*vx)/np.sqrt(r**2-z**2)
    vtheta = (z/(r*np.sqrt(r**2-z**2)))*(x*vx + y*vy + ((z/r)**2-1)*vz)
    return vr,vphi,vtheta

vr,vphi,vtheta = v_sph(Rdist,xrel,yrel,zrel,vx_obs,vy_obs,vz_obs)

nbeta=10  # set number of beta parameters to use

beta_obs=np.zeros(nbeta)

for i in range(0,nbeta):
    indx = np.where(Rdist[1:]<=Rmax/nbeta*(i+1))
    beta_obs [i] = 1 - (np.mean(vphi[indx]**2)+np.mean(vtheta**2))/(2*np.mean(vr**2))

print('\nM200 (Msun): ',M200,' R200 (Msun): ',R200)
print('Anisotropy parameters (from R=0 to Rmax): ',beta_obs)