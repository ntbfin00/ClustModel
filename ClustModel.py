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

# Read in observations from file and add appropriate units
obs = table.QTable.read(filepath + "obs.dat", format="ascii", names=["x","y","vz","sigz","M","R","zrel","vr"]) 
obs["x"].unit = u.arcsec
obs["y"].unit = u.arcsec
obs["vz"].unit = u.km/u.s
obs["sigz"].unit = u.km/u.s

Rdist = obs['R']
Rmax = np.max(Rdist)
zrel = obs['zrel']

# Read in cluster parameters from file and add appropriate units
par = table.QTable.read(filepath + "clust_params.dat", format="ascii", names=["M200","R200","beta"])  


#======================SET PARAMETERS TO FIT===========================

# Use par table to use pre-determined parameters
c_par=10  # set NFW concentration parameter
M200=par['M200'][0]  # set cluster halo M200 value (Msun)
R200=par['R200'][0]  # set cluster halo R200 value (pc)
beta=par['beta']  # set cluster anisotropy parameters as array


#======================TRACER DENSITY FIT==============================

#Determine tracer number density distribution (logarithmically binned)
bins=20  # number of bins
tracer_hist = np.histogram(np.log(Rdist[1:]),bins)  # histogram data

# histogram of tracers
plt.hist(np.log(Rdist[1:]),bins)
plt.title('Histogram of tracers')
plt.xlabel('ln(R) (pc)')
plt.ylabel('Count')
plt.show()

# histogram of tracer number density
shell_vol = np.zeros(bins)
density = np.zeros(bins)
logr = np.zeros(bins)

for i in range(0,bins):
    shell_vol[i] = (4/3)*np.pi*(np.exp(tracer_hist[1][i+1])**3-np.exp(tracer_hist[1][i])**3)  # pc^3
    density[i] = tracer_hist[0][i]/shell_vol[i]   # pc^(-3)
    logr[i] = (tracer_hist[1][i+1]-tracer_hist[1][i])/2+tracer_hist[1][i]

plt.bar(logr,density)
plt.title('Tracerumber density distribution')
plt.xlabel('ln(R) (pc)')
plt.ylabel('Number density $(pc^{-3})$')
plt.yscale("log")
plt.show()

plt.plot(np.exp(logr)/1e6,density,marker = 'o')
plt.yscale('log')
plt.title('Tracer number density (logarithmic sampling)')
plt.xlabel('R $(10^6 pc)$')
plt.ylabel('Number density $(pc^{-3})$')
plt.show()

# Fit polynomial
no_val = np.where(density == 0)[0]  # disregard zero values that will mess with the polynomial fit
r_fit = np.delete(logr, no_val)
dens_fit = np.delete(density, no_val)

xnew = np.linspace(np.min(logr), np.max(logr),1000)

deg = 5  # chosen polynomial degree
a = np.polyfit(r_fit,np.log(dens_fit),deg)

def poly(x,a,deg):
    p=0
    for i in range(0,deg+1):
        n = deg - i
        p = p + a[i]*x**n
    return p

plt.plot(xnew,poly(xnew,a,deg),linewidth='3')
plt.plot(logr,np.log(density), marker='o')
plt.title("Tracer number density fit")
plt.ylabel(r"$\ln{\nu}\,\, (pc^{-3})$")
plt.xlabel("ln(R) (pc)")
plt.show()


#======================DARK MATTER NFW DENSITY PROFILE=======================

def nfw(r,M200,R200,c):  # Msun/pc^3
    Rs=R200/c  # in pc
    A_nfw = np.log(1+c) - c/(1+c)
    
    nfw = M200/(4*np.pi*Rs**3*A_nfw*(r/Rs)*(1 + (r/Rs))**2)  

    return nfw

halo_ext = 1.2*Rmax  # set extent of NFW profile in terms of distance of furthest tracer

r = np.geomspace(1, halo_ext, 300)  # logarithmically spaced radii in pc
y = nfw(r,M200,R200,c_par)  # The profile must be logarithmically sampled!

# Plot NFW profile
plt.title('NFW density profile')
plt.loglog(r,y)
plt.xlabel('Radius  (pc)')
plt.ylabel(''r'$\rho\,\,(Msun/pc^3)$')
plt.show()


#===========================JEANS MODEL===================================

def lognu(r):  # natural logarithm of tracer density in pc^-3
    return poly(np.log(r),a,deg)

def submass(r):  # determine stellar mass enclosed at radius r
    objs = np.where(Rdist<r)
    return np.sum(obs['M'][objs])

# Spherical Jeans equation (1st order differential equation)
def jeansODE(vr,r,beta):
    bindx = int(np.floor(r/Rmax*len(beta)))  # index to determine which beta value to use
    if bindx>=len(beta):  # condition to ensure approximate solution doesn't breakdown
        bindx=len(beta)-1
    
    rho = lambda r: r**2*nfw(r,M200,R200,c_par)  # integrate to obtain mass profile
    m_nfw = (4*np.pi*integrate.quad(rho, 0, r)[0])
  
    dvdr = -vr*(2*beta[bindx]/r + derivative(lognu, r, dx=1e-30))-(G/r**2)*(submass(r)+m_nfw)
    return dvdr


vr0 = 0 # initial condition at cluster centre r=0 (arbitrary)
nbins = 1000  # set number of velocity bins for the model
rbin = np.linspace(1, Rmax, nbins)    # Rmin=1 to avoid division by 0

# Solve ODE for radial velocity
vr_rms = np.sqrt(abs(odeint(jeansODE, vr0, rbin,args=(beta,))))   # in km/s

# Plot the rotation curve
plt.plot(rbin,vr_rms)
plt.title('Rotation curve')
plt.ylabel('Absolute radial v_rms (km/s)')
plt.xlabel('Clustocentric radius (pc)')
plt.xscale('log')
plt.show()

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
    
# Write model velocities and dispersions to file
units = 'vz (km/s)   sigma_z (km/s)'
data = np.column_stack((vz_mod,sigma_mod))

np.savetxt(filepath + 'model.dat',data, header = units)

# read in the observed velocities from file and add appropriate units
mod = table.QTable.read(filepath + "model.dat", format="ascii", names=["vz","sigz"])  # observing along z
mod["vz"].unit = u.km/u.s
mod["sigz"].unit = u.km/u.s

# Plot observations vs. model
col_lim = 2000  # limit of colorbar

f1, ax = plt.subplots(2,3,sharex='all', sharey='all',figsize=(15,8))
f1.tight_layout()
f1.show()

# Observed line of sight velocity and dispersion maps
obs1 = ax[0,0].scatter(obs["x"]*1e3,obs["y"]*1e3,c=obs['vz'],cmap='jet',s=9)
ax[0,0].set_title('observed $(km\,s^{-1})$',fontsize=25)
ax[0,0].set_ylabel('y ($10^{-3}$ arcsec)',fontsize=20)
ax[0,0].text(0.06, 0.88, '$v_z$', style='oblique', transform=ax[0,0].transAxes,fontsize=27)
obs1.set_clim(-col_lim, col_lim)
plt.colorbar(obs1,ax=ax[0,0])

obs2 = ax[1,0].scatter(obs["x"]*1e3,obs["y"]*1e3,c=obs['sigz'],cmap='jet',s=9)
ax[1,0].set_xlabel('x ($10^{-3}$ arcsec)',fontsize=20)
ax[1,0].set_ylabel('y ($10^{-3}$ arcsec)',fontsize=20)
ax[1,0].text(0.06, 0.88, '$\sigma_z$', style='oblique', transform=ax[1,0].transAxes,fontsize=27)
obs2.set_clim(0, col_lim*(3/4))
plt.colorbar(obs2,ax=ax[1,0])

# Model line of sight velocity and dispersion maps
mod1 = ax[0,1].scatter(obs["x"]*1e3,obs["y"]*1e3,c=mod['vz'],cmap='jet',s=9)
ax[0,1].set_title('model $(km\,s^{-1})$',fontsize=25)
ax[0,1].text(0.06, 0.88, '$v_z$', style='oblique', transform=ax[0,1].transAxes,fontsize=27)
mod1.set_clim(-col_lim, col_lim)
plt.colorbar(mod1,ax=ax[0,1])

mod2 = ax[1,1].scatter(obs["x"]*1e3,obs["y"]*1e3,c=mod['sigz'],cmap='jet',s=9)
ax[1,1].set_xlabel('x ($10^{-3}$ arcsec)',fontsize=20)
ax[1,1].text(0.06, 0.88, '$\sigma_z$', style='oblique', transform=ax[1,1].transAxes,fontsize=27)
mod2.set_clim(0, col_lim*(3/4))
plt.colorbar(mod2,ax=ax[1,1])

# Residual plots
res1 = ax[0,2].scatter(obs["x"]*1e3,obs["y"]*1e3,c=(mod['vz']-obs['vz']),cmap='jet',s=9)
ax[0,2].set_title('residual $(km\,s^{-1})$',fontsize=25)
ax[0,2].text(0.06, 0.88, '$v_z$', style='oblique', transform=ax[0,2].transAxes,fontsize=27)
res1.set_clim(-col_lim, col_lim)
plt.colorbar(res1,ax=ax[0,2])

res2 = ax[1,2].scatter(obs["x"]*1e3,obs["y"]*1e3,c=(mod['sigz']-obs['sigz']),cmap='jet',s=9)
ax[1,2].set_xlabel('x ($10^{-3}$ arcsec)',fontsize=20)
ax[1,2].text(0.06, 0.88, '$\sigma_z$', style='oblique', transform=ax[1,2].transAxes,fontsize=27)
res2.set_clim(-col_lim/2, col_lim/2)
plt.colorbar(res2,ax=ax[1,2])
