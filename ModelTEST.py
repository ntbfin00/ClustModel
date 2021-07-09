#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 10:30:58 2021

@author: nathanfindlay
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
from astropy import table, units as u
from scipy.integrate import odeint
from scipy import integrate
from scipy.misc import derivative

filepath='/Users/nathanfindlay/Gals_SummerProject/data_files/'#'/YOUR_FILEPATH'  # path to where input files are stored

# Define astronomical constants
G = 4.3009125e-3 # (km/s)^2 pc/Msun
redshift=0
scale_factor = 1.0 / (1+redshift)
h = 0.6774 # choose your H0 
H0 = h*100 # km/Mpc/s
cosmo = FlatLambdaCDM(H0=H0, Om0=0.2726)
rho_crit = (3*(H0/1e6)**2)/(8*np.pi*G)  # Msun/pc^3

# Read in observations from file and add appropriate units
obs = table.QTable.read(filepath + "obs.dat", format="ascii", names=["x","y","vz","sigz","M","R","zrel","vr","vphi","vtheta"]) 
obs["x"].unit = u.arcsec
obs["y"].unit = u.arcsec
obs["vz"].unit = u.km/u.s
obs["sigz"].unit = u.km/u.s

Rdist = obs['R']
Rmax = np.max(Rdist)
Rmin = np.min(Rdist[1:])
zrel = obs['zrel']

# Read in cluster parameters from file and add appropriate units
#par = table.QTable.read(filepath + "clust_params.dat", format="ascii", names=["M200","R200","beta"])  
par = table.QTable.read(filepath + "clust_params.dat", format="ascii", names=["M200","R200"])  

def beta_obs(R200,cNFW):  # observed anisotropy parameters either side of critical radius
    Rs=R200/cNFW  # determine scale radius Rs
    
    less_Rs = np.where(Rdist[1:]<Rs)[0]  # split either side
    more_Rs = np.where(Rdist[1:]>=Rs)[0]
    split = [less_Rs, more_Rs]
    b = np.zeros(2)
    for i in range(0,2):
        r = split[i]
        b[i] = 1 - (np.mean(obs['vphi'][1:][r]**2)+np.mean(obs['vtheta'][1:][r]**2))/(2*np.mean(obs['vr'][1:][r]**2))
    return b

#======================SET PARAMETERS TO FIT===========================

# Use par table to use pre-determined Illustris parameters
cNFW = 18  # set NFW concentration parameter
M200 = par['M200'][0]  # set cluster halo M200 value (Msun)
R200 = par['R200'][0]  # set cluster halo R200 value (pc)
# If R200 not pre-determined use R200 = (3*M200/(4*np.pi*200*rho_crit))**(1/3)

#Rs=R200/cNFW  # scale radius in pc

beta = beta_obs(R200,cNFW)  # set cluster anisotropy parameters

print('ILLUSTRIS PARAMETERS:\nc =',cNFW,'| M200 =',M200,'| R200 =',R200,'| beta =',beta,'\n')

#=========================VARY PARAMETERS===============================
novar_par = cNFW,M200,beta,R200  # unvaried parameters

#======================TRACER DENSITY FIT==============================

#Determine tracer number density distribution (logarithmically binned)
bins=20  # number of bins
tracer_hist = np.histogram(np.log(Rdist[1:]),bins)  # histogram data

# histogram of tracer number density
shell_vol = np.zeros(bins)
density = np.zeros(bins)
logr = np.zeros(bins)

for i in range(0,bins):
    shell_vol[i] = (4/3)*np.pi*(np.exp(tracer_hist[1][i+1])**3-np.exp(tracer_hist[1][i])**3)  # pc^3
    density[i] = tracer_hist[0][i]/shell_vol[i]   # pc^(-3)
    logr[i] = (tracer_hist[1][i+1]-tracer_hist[1][i])/2+tracer_hist[1][i]
    

# Fit polynomial
no_val = np.where(density == 0)[0]  # disregard zero values that will mess with the polynomial fit
r_fit = np.delete(logr, no_val)
dens_fit = np.delete(density, no_val)

deg = 5  # chosen polynomial degree
a = np.polyfit(r_fit,np.log(dens_fit),deg)

def poly(x,a,deg):
    p=0
    for i in range(0,deg+1):
        n = deg - i
        p = p + a[i]*x**n
        return p
        

#======================DARK MATTER NFW DENSITY PROFILE=======================

def nfw(r,M200,R200,c):  # Msun/pc^3
    Rs=R200/c  # in pc
    A_nfw = np.log(1+c) - c/(1+c)
    
    nfw = M200/(4*np.pi*Rs**3*A_nfw*(r/Rs)*(1 + (r/Rs))**2)  

    return nfw

halo_ext = 1.2*Rmax  # set extent of NFW profile in terms of distance of furthest tracer


#===========================JEANS MODEL===================================

def lognu(r):  # natural logarithm of tracer density in pc^-3
    return poly(np.log(r),a,deg)

def submass(r):  # determine stellar mass enclosed at radius r
    objs = np.where(Rdist<r)
    return np.sum(obs['M'][objs])

# Spherical Jeans equation (1st order differential equation)
def jeansODE(vr,r,beta,M200,R200,cNFW):
    rc=R200/cNFW
    if r<Rmin:
        B=1
    elif r<rc:
        B=beta[0]
    else:
        B=beta[1]
    
    rho = lambda r: r**2*nfw(r,M200,R200,cNFW)  # integrate to obtain mass profile
    m_nfw = (4*np.pi*integrate.quad(rho, 0, r)[0])
  
    dvdr = -vr*(2*B/r + derivative(lognu, r, dx=1e-30))-(G/r**2)*(submass(r)+m_nfw)

    return dvdr

vr0 = 0  # initial condition at cluster centre r=0 (arbitrary)
nbins = 1000  # set number of velocity bins for the model
rbin = np.linspace(1, Rmax, nbins)    # Rmin=1 to avoid division by 0
'''
def vary_param(factor):  # vary parameters by a chosen factor
    c_var = factor*cNFW  
    M200_var = factor*M200   
    R200_var = ((factor)**(1/3))*R200
    beta_var = factor*beta_obs(R200,cNFW)  # set cluster anisotropy parameters
    return c_var, M200_var, beta_var, R200_var
'''
'''
def vary_param(percent):  # vary parameters by a chosen factor
    factor = 1+(percent/100)
    log_factor = 10**(percent/10)
    c_var = factor*cNFW  
    M200_var = (log_factor)*M200   
    R200_var = (log_factor**(1/3))*R200
    beta_var = factor*beta_obs(R200,cNFW)  # set cluster anisotropy parameters
    return c_var, M200_var, beta_var, R200_var 

vels = np.zeros((9,len(Rdist)))
disps = np.zeros((9,len(Rdist)))

percent = 10  # set percentage to vary parameter
min_par = vary_param(-percent)  # parameters -percent
novar_par = vary_param(0)  # unchanged parameters
plus_par = vary_param(percent)  # parameters +percent

test_par = np.concatenate((min_par[:-1],plus_par[:-1]))  # parameters to test

for n in range(0,7):
    cNFW, M200, beta, R200 = novar_par
    
    if (n==1) or (n==4):
        cNFW = test_par[n-1]
    if (n==2):
        M200 = test_par[n-1]
        R200 = min_par[3]
    if (n==5):
        M200 = test_par[n-1]
        R200 = plus_par[3]
    if (n==3) or (n==6):
        beta = test_par[n-1] 
    else:
        beta = beta_obs(R200,cNFW)

    print('RUN:',n+1,'| c =',cNFW,'| M200 =',M200,'| beta =',beta)

#======================TRACER DENSITY FIT==============================

    #Determine tracer number density distribution (logarithmically binned)
    bins=20  # number of bins
    tracer_hist = np.histogram(np.log(Rdist[1:]),bins)  # histogram data

    # histogram of tracer number density
    shell_vol = np.zeros(bins)
    density = np.zeros(bins)
    logr = np.zeros(bins)

    for i in range(0,bins):
        shell_vol[i] = (4/3)*np.pi*(np.exp(tracer_hist[1][i+1])**3-np.exp(tracer_hist[1][i])**3)  # pc^3
        density[i] = tracer_hist[0][i]/shell_vol[i]   # pc^(-3)
        logr[i] = (tracer_hist[1][i+1]-tracer_hist[1][i])/2+tracer_hist[1][i]
    

    # Fit polynomial
    no_val = np.where(density == 0)[0]  # disregard zero values that will mess with the polynomial fit
    r_fit = np.delete(logr, no_val)
    dens_fit = np.delete(density, no_val)

    #xnew = np.linspace(np.min(logr), np.max(logr),1000)

    deg = 5  # chosen polynomial degree
    a = np.polyfit(r_fit,np.log(dens_fit),deg)

    def poly(x,a,deg):
        p=0
        for i in range(0,deg+1):
            n = deg - i
            p = p + a[i]*x**n
            return p
        

#======================DARK MATTER NFW DENSITY PROFILE=======================

    def nfw(r,M200,R200,c):  # Msun/pc^3
        Rs=R200/c  # in pc
        A_nfw = np.log(1+c) - c/(1+c)
    
        nfw = M200/(4*np.pi*Rs**3*A_nfw*(r/Rs)*(1 + (r/Rs))**2)  

        return nfw

    halo_ext = 1.2*Rmax  # set extent of NFW profile in terms of distance of furthest tracer


#===========================JEANS MODEL===================================

    def lognu(r):  # natural logarithm of tracer density in pc^-3
        return poly(np.log(r),a,deg)

    def submass(r):  # determine stellar mass enclosed at radius r
        objs = np.where(Rdist<r)
        return np.sum(obs['M'][objs])

    # Spherical Jeans equation (1st order differential equation)
    def jeansODE(vr,r,beta,M200,R200,cNFW):
        rc=R200/cNFW
        if r<Rmin:
            B=1
        elif r<rc:
            B=beta[0]
        else:
            B=beta[1]
    
        rho = lambda r: r**2*nfw(r,M200,R200,cNFW)  # integrate to obtain mass profile
        m_nfw = (4*np.pi*integrate.quad(rho, 0, r)[0])
  
        dvdr = -vr*(2*B/r + derivative(lognu, r, dx=1e-30))-(G/r**2)*(submass(r)+m_nfw)

        return dvdr

    vr0 = 0  # initial condition at cluster centre r=0 (arbitrary)
    nbins = 1000  # set number of velocity bins for the model
    rbin = np.linspace(1, Rmax, nbins)    # Rmin=1 to avoid division by 0
'''
'''
    # Solve ODE for radial velocity
    vr_rms = np.sqrt(abs(odeint(jeansODE, vr0**2, rbin,args=(beta,M200,R200,cNFW))))   # in km/s

    vr_mod=np.zeros(len(Rdist))
    vz_mod=np.zeros(len(Rdist))
    sigma_mod=np.zeros(len(Rdist))
    gamma=np.zeros(len(Rdist))

    rbin_indx = np.digitize(Rdist, rbin, right=True)

    for i in range(1,len(Rdist)):
        vz_mod[0] = 0  # add primary subhalo vz
        vr_mod[i] = vr_rms[rbin_indx[i]]  # sets abs(vr) for each object based on Jeans solution
        if obs['vr'][i]<0:  # sets vr direction from observed vr
            sign = -1
        else:
            sign = 1
        vz_mod[i] = sign*vr_mod[i]*(zrel[i]/Rdist[i])  # vz=vr.cos(theta)=vr.(z/r)
    

    # Calculate dispersions by binning velocities in radius
    nsig_bin = 20  # number of velocity bins for dispersion calculations (same should be used in TNGextract)
    sig_bin = np.linspace(0, Rmax, nsig_bin)  # make evenly distributed bins
    sigbin_indx = np.digitize(Rdist, sig_bin, right=True)
    for i in range(0,len(rbin)):
        bin_objs = np.where(sigbin_indx == i)[0]  # determine what tracers lie in each annulus
        v_ms = np.mean(vz_mod[bin_objs]**2)  # mean of the square velocities
        v_sm = np.mean(vz_mod[bin_objs])**2  # square of the mean velocities
        sigma_mod[bin_objs] = np.sqrt(v_ms - v_sm)  # set sigma for each tracer in annulus

    vels[n] = vz_mod
    disps[n] = sigma_mod


#==========================CREATE PLOTS===========================

col_lim = 1500  # limit of colorbar


rows = ['c','$M_{200}$',r'$\beta$']

f1, axes= plt.subplots(4,2,sharex='all', sharey='all',figsize=(11,14))

pad = 50 # in points

for ax, row in zip(axes[1:,0], rows):
    ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size=27, ha='center', va='center',style='italic')

f1.tight_layout()
f1.subplots_adjust(left=0.15, top=0.95)

# Observed map
observed = axes[0,0].scatter(obs["x"]*1e3,obs["y"]*1e3,c=obs['sigz'],cmap='jet',s=9)
axes[0,0].set_title('observed $(km\,s^{-1})$',fontsize=25)
axes[0,0].set_ylabel('y ($10^{-3}$ arcsec)',fontsize=20)
axes[0,0].text(0.06, 0.88, '$\sigma_z$', style='oblique', transform=axes[0,0].transAxes,fontsize=27)
observed.set_clim(0, col_lim)

# Model fit to Illustris parameters 
best = axes[0,1].scatter(obs["x"]*1e3,obs["y"]*1e3,c=disps[0],cmap='jet',s=9)
axes[0,1].set_title('model fit $(km\,s^{-1})$',fontsize=25)
axes[0,1].text(0.06, 0.88, '$\sigma_z$', style='oblique', transform=axes[0,1].transAxes,fontsize=27)
best.set_clim(0, col_lim)

# Model -percent maps
minC = axes[1,0].scatter(obs["x"]*1e3,obs["y"]*1e3,c=disps[1],cmap='jet',s=9)
axes[1,0].set_title('model -'+str(percent)+'% $(km\,s^{-1})$',fontsize=25)
axes[1,0].set_ylabel('y ($10^{-3}$ arcsec)',fontsize=20)
axes[1,0].text(0.06, 0.88, '$\sigma_z$', style='oblique', transform=axes[1,0].transAxes,fontsize=27)
minC.set_clim(0, col_lim)
plt.colorbar(minC,ax=axes[1,0])

minM = axes[2,0].scatter(obs["x"]*1e3,obs["y"]*1e3,c=disps[2],cmap='jet',s=9)
axes[2,0].set_ylabel('y ($10^{-3}$ arcsec)',fontsize=20)
axes[2,0].text(0.06, 0.88, '$\sigma_z$', style='oblique', transform=axes[2,0].transAxes,fontsize=27)
minM.set_clim(0, col_lim)
plt.colorbar(minM,ax=axes[2,0])

minB = axes[3,0].scatter(obs["x"]*1e3,obs["y"]*1e3,c=disps[3],cmap='jet',s=9)
axes[3,0].set_xlabel('x ($10^{-3}$ arcsec)',fontsize=20)
axes[3,0].set_ylabel('y ($10^{-3}$ arcsec)',fontsize=20)
axes[3,0].text(0.06, 0.88, '$\sigma_z$', style='oblique', transform=axes[3,0].transAxes,fontsize=27)
minB.set_clim(0, col_lim)
plt.colorbar(minB,ax=axes[3,0])

# Model +percent maps
plusC = axes[1,1].scatter(obs["x"]*1e3,obs["y"]*1e3,c=disps[4],cmap='jet',s=9)
axes[1,1].set_title('model +'+str(percent)+'% $(km\,s^{-1})$',fontsize=25)
axes[1,1].text(0.06, 0.88, '$\sigma_z$', style='oblique', transform=axes[1,1].transAxes,fontsize=27)
plusC.set_clim(0, col_lim)
plt.colorbar(plusC,ax=axes[1,1])

plusM = axes[2,1].scatter(obs["x"]*1e3,obs["y"]*1e3,c=disps[5],cmap='jet',s=9)
axes[2,1].text(0.06, 0.88, '$\sigma_z$', style='oblique', transform=axes[2,1].transAxes,fontsize=27)
plusM.set_clim(0, col_lim)
plt.colorbar(plusM,ax=axes[2,1])

plusB = axes[3,1].scatter(obs["x"]*1e3,obs["y"]*1e3,c=disps[6],cmap='jet',s=9)
axes[3,1].text(0.06, 0.88, '$\sigma_z$', style='oblique', transform=axes[3,1].transAxes,fontsize=27)
axes[3,1].set_xlabel('x ($10^{-3}$ arcsec)',fontsize=20)
plusB.set_clim(0, col_lim)
plt.colorbar(plusB,ax=axes[3,1])

box1 = axes[0,0].get_position()
box2 = axes[0,1].get_position()
box1.x1 = box1.x1 - 0.08
box1.y0 = box1.y0 + 0.03
box1.y1 = box1.y1 + 0.03
box2.x1 = box2.x1 - 0.08
box2.y0 = box2.y0 + 0.03
box2.y1 = box2.y1 + 0.03
axes[0,0].set_position(box1,which='both')
axes[0,1].set_position(box2,which='both')

cax1 = f1.add_axes([box1.x1+0.02,box1.y0,0.014,box1.y1-box1.y0])
cax2 = f1.add_axes([box2.x1+0.02,box2.y0,0.014,box2.y1-box2.y0])

plt.colorbar(observed,cax=cax1)
plt.colorbar(best,cax=cax2)

plt.show()


#=======================PLOT RESIDUALS===========================

f2, axes= plt.subplots(4,2,sharex='all', sharey='all',figsize=(11,14))

pad = 50 # in points

for ax, row in zip(axes[1:,0], rows):
    ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size=27, ha='center', va='center',style='italic')

f2.tight_layout()
f2.subplots_adjust(left=0.15, top=0.95)

# Observed map
observed = axes[0,0].scatter(obs["x"]*1e3,obs["y"]*1e3,c=obs['sigz'],cmap='jet',s=9)
axes[0,0].set_title('observed $(km\,s^{-1})$',fontsize=25)
axes[0,0].set_ylabel('y ($10^{-3}$ arcsec)',fontsize=20)
axes[0,0].text(0.06, 0.88, '$\sigma_z$', style='oblique', transform=axes[0,0].transAxes,fontsize=27)
observed.set_clim(0, col_lim)

# Model fit to Illustris parameters 
best = axes[0,1].scatter(obs["x"]*1e3,obs["y"]*1e3,c=(disps[0]*u.km/u.s-obs['sigz']),cmap='jet',s=9)
axes[0,1].set_title('model residual $(km\,s^{-1})$',fontsize=25)
axes[0,1].text(0.06, 0.88, '$\sigma_z$', style='oblique', transform=axes[0,1].transAxes,fontsize=27)
best.set_clim(-col_lim/2, col_lim/2)

# Model -percent maps
minC = axes[1,0].scatter(obs["x"]*1e3,obs["y"]*1e3,c=(disps[1]*u.km/u.s-obs['sigz']),cmap='jet',s=9)
axes[1,0].set_title('residual -'+str(percent)+'% $(km\,s^{-1})$',fontsize=25)
axes[1,0].set_ylabel('y ($10^{-3}$ arcsec)',fontsize=20)
axes[1,0].text(0.06, 0.88, '$\sigma_z$', style='oblique', transform=axes[1,0].transAxes,fontsize=27)
minC.set_clim(-col_lim/2, col_lim/2)
plt.colorbar(minC,ax=axes[1,0])

minM = axes[2,0].scatter(obs["x"]*1e3,obs["y"]*1e3,c=(disps[2]*u.km/u.s-obs['sigz']),cmap='jet',s=9)
axes[2,0].set_ylabel('y ($10^{-3}$ arcsec)',fontsize=20)
axes[2,0].text(0.06, 0.88, '$\sigma_z$', style='oblique', transform=axes[2,0].transAxes,fontsize=27)
minM.set_clim(-col_lim/2, col_lim/2)
plt.colorbar(minM,ax=axes[2,0])

minB = axes[3,0].scatter(obs["x"]*1e3,obs["y"]*1e3,c=(disps[3]*u.km/u.s-obs['sigz']),cmap='jet',s=9)
axes[3,0].set_xlabel('x ($10^{-3}$ arcsec)',fontsize=20)
axes[3,0].set_ylabel('y ($10^{-3}$ arcsec)',fontsize=20)
axes[3,0].text(0.06, 0.88, '$\sigma_z$', style='oblique', transform=axes[3,0].transAxes,fontsize=27)
minB.set_clim(-col_lim/2, col_lim/2)
plt.colorbar(minB,ax=axes[3,0])

# Model +percent maps
plusC = axes[1,1].scatter(obs["x"]*1e3,obs["y"]*1e3,c=(disps[4]*u.km/u.s-obs['sigz']),cmap='jet',s=9)
axes[1,1].set_title('residual +'+str(percent)+'% $(km\,s^{-1})$',fontsize=25)
axes[1,1].text(0.06, 0.88, '$\sigma_z$', style='oblique', transform=axes[1,1].transAxes,fontsize=27)
plusC.set_clim(-col_lim/2, col_lim/2)
plt.colorbar(plusC,ax=axes[1,1])

plusM = axes[2,1].scatter(obs["x"]*1e3,obs["y"]*1e3,c=(disps[5]*u.km/u.s-obs['sigz']),cmap='jet',s=9)
axes[2,1].text(0.06, 0.88, '$\sigma_z$', style='oblique', transform=axes[2,1].transAxes,fontsize=27)
plusM.set_clim(-col_lim/2, col_lim/2)
plt.colorbar(plusM,ax=axes[2,1])

plusB = axes[3,1].scatter(obs["x"]*1e3,obs["y"]*1e3,c=(disps[6]*u.km/u.s-obs['sigz']),cmap='jet',s=9)
axes[3,1].text(0.06, 0.88, '$\sigma_z$', style='oblique', transform=axes[3,1].transAxes,fontsize=27)
axes[3,1].set_xlabel('x ($10^{-3}$ arcsec)',fontsize=20)
plusB.set_clim(-col_lim/2, col_lim/2)
plt.colorbar(plusB,ax=axes[3,1])

box1 = axes[0,0].get_position()
box2 = axes[0,1].get_position()
box1.x1 = box1.x1 - 0.08
box1.y0 = box1.y0 + 0.03
box1.y1 = box1.y1 + 0.03
box2.x1 = box2.x1 - 0.08
box2.y0 = box2.y0 + 0.03
box2.y1 = box2.y1 + 0.03
axes[0,0].set_position(box1,which='both')
axes[0,1].set_position(box2,which='both')

cax1 = f2.add_axes([box1.x1+0.02,box1.y0,0.014,box1.y1-box1.y0])
cax2 = f2.add_axes([box2.x1+0.02,box2.y0,0.014,box2.y1-box2.y0])

plt.colorbar(observed,cax=cax1)
plt.colorbar(best,cax=cax2)

plt.show()
'''

#==================RESIDUAL AVERAGE WITH VARYING PARAMETER=================

steps = 50  # set number of steps to vary parameters by 
mean_err = np.zeros((4,steps))
c_var = np.linspace(0.1,40,steps)
M_var = np.geomspace(1e9,1e22,steps)
B_var = np.linspace(-1,1,steps)

for j in range(0,4):  # vary c, M200 and both beta parameters seperately
    par_name = ['c','M200','Beta 1','Beta 2']
    print('Varying '+str(par_name[j])+' parameter...')
    for n in range(0,steps):
        cNFW, M200, beta, R200 = novar_par  # set to unvaried values
        if j==0:
            cNFW = c_var[n]
        if j==1:
            M200 = M_var[n]
            R200 = novar_par[3]*(M200/novar_par[1])  # vary R200 proportionally 
        if j==2:
            beta = [B_var[n],beta[1]]
        if j==3:
            beta = [beta[0],B_var[n]]

        # Solve ODE for radial velocity
        vr_rms = np.sqrt(abs(odeint(jeansODE, vr0**2, rbin,args=(beta,M200,R200,cNFW))))  # in km/s

        vr_mod=np.zeros(len(Rdist))
        vz_mod=np.zeros(len(Rdist))
        sigma_mod=np.zeros(len(Rdist))
        gamma=np.zeros(len(Rdist))

        rbin_indx = np.digitize(Rdist, rbin, right=True)
        for i in range(1,len(Rdist)):
            vz_mod[0] = 0  # add primary subhalo vz
            vr_mod[i] = vr_rms[rbin_indx[i]]  # sets abs(vr) for each object based on Jeans solution
            if obs['vr'][i]<0:  # sets vr direction from observed vr
                sign = -1
            else:
                sign = 1
            vz_mod[i] = sign*vr_mod[i]*(zrel[i]/Rdist[i])  # vz=vr.cos(theta)=vr.(z/r)
    

        # Calculate dispersions by binning velocities in radius
        nsig_bin = 20  # number of velocity bins for dispersion calculations (same should be used in TNGextract)
        sig_bin = np.linspace(0, Rmax, nsig_bin)  # make evenly distributed bins
        sigbin_indx = np.digitize(Rdist, sig_bin, right=True)
        for i in range(0,len(rbin)):
            bin_objs = np.where(sigbin_indx == i)[0]  # determine what tracers lie in each annulus
            v_ms = np.mean(vz_mod[bin_objs]**2)  # mean of the square velocities
            v_sm = np.mean(vz_mod[bin_objs])**2  # square of the mean velocities
            sigma_mod[bin_objs] = np.sqrt(v_ms - v_sm)  # set sigma for each tracer in annulus

        # Calculate average residual percentage 
        prcnt_sig = abs((sigma_mod*u.km/u.s-obs['sigz'])/obs['sigz'])*100
        prcnt_sig = [x for x in prcnt_sig if np.isnan(x) == False]  # remove nan values
        mean_err[j][n] = np.mean(prcnt_sig)

# Set for each parameter
c_err = mean_err[0]
M_err = mean_err[1]
B1_err = mean_err[2] 
B2_err = mean_err[3]

# Plot average residual with parameter value
col_lim = 1500  # limit of colorbar
rows = ['c','$M_{200}$',r'$\beta_{<R_s}$',r'$\beta_{>R_s}$']
pad = 50 # in points

f3, ((ax1,ax2,ax3,ax4))= plt.subplots(4,1,figsize=((7,12)))

axes = ((ax1,ax2,ax3,ax4))
for ax, row in zip(axes[0:], rows):
    ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size=27, ha='center', va='center',style='italic')
    
f3.tight_layout(pad=2.7)
f3.show()

ax1.plot(c_var,c_err)
ax1.axvline(x=novar_par[0],linestyle='--',color='g',label='Visual fit')
#ax1.set_xlabel('c parameter',fontsize=15)
ax1.set_ylabel('$(\sigma_{mod}-\sigma_{obs})/\sigma_{obs}\,\, (\%)$',fontsize=15)
ax1.legend()

ax2.plot(M_var,M_err)
ax2.axvline(x=novar_par[1],linestyle='--',color='r',label='Illustris value')
#ax2.set_xlabel('$M_{200}$ parameter',fontsize=15)
ax2.set_ylabel('$(\sigma_{mod}-\sigma_{obs})/\sigma_{obs}\,\, (\%)$',fontsize=15)
ax2.set_xscale('log')
ax2.legend()

ax3.plot(B_var,B1_err)
ax3.axvline(x=novar_par[2][0],linestyle='--',color='r',label='Illustris value')
#ax3.set_xlabel(r'$\beta$'+' interior to $R_s$',fontsize=15)
ax3.set_ylabel('$(\sigma_{mod}-\sigma_{obs})/\sigma_{obs}\,\, (\%)$',fontsize=15)
ax3.set_yscale('log')
ax3.legend()

ax4.plot(B_var,B2_err)
ax4.axvline(x=novar_par[2][1],linestyle='--',color='r',label='Illustris value')
#ax4.set_xlabel(r'$\beta$'+' exterior to $R_s$',fontsize=15)
ax4.set_ylabel('$(\sigma_{mod}-\sigma_{obs})/\sigma_{obs}\,\, (\%)$',fontsize=15)
ax4.set_yscale('log')
ax4.legend()

f3.suptitle("Group "+str(grnr),y=1,x=0.6,fontsize=20)
