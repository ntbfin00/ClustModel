#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 09:01:25 2021

@author: nathanfindlay
"""
import numpy as np
import matplotlib.pyplot as plt
import requests
#import illustris_python as il
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
        filename = cutoutsPath + r.headers['content-disposition'].split("filename=")[1]
        with open(filename, 'wb') as f:
            f.write(r.content)
        return filename # return the filename string

    return r

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
xrel = xrel[xrel != 0]  # exclude position of central subhalo
yrel = ypos-yc
yrel = yrel[yrel != 0]
zrel = zpos-zc
zrel = zrel[zrel != 0]

# Calculate projected tracer positions in arcsecs
xproj = np.arctan((xrel)/(Dist+(zrel)))
yproj = np.arctan((yrel)/(Dist+(zrel)))

# Write to file
units = 'x\' (arcsecs)   y\' (arcsecs)'
data = np.column_stack((xproj,yproj))  # in pc, no z as observing along z axis

np.savetxt(datapath + 'positions.dat', data, header=units)
print('Positions saved to file')

# Read in positions from file and add appropriate units
pos = table.QTable.read(datapath + "positions.dat", format="ascii", names=["x","y"])  
pos["x"].unit = u.arcsec
pos["y"].unit = u.arcsec
  
# Plot positions  
plt.scatter(pos["x"],pos["y"],s=9)
plt.title('Group '+str(grnr),fontsize=20)
plt.xlabel('x (arcsec)',fontsize=20)
plt.ylabel('y (arcsec)',fontsize=20)
plt.show()
