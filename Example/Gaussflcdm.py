from numpy import pi, log, sqrt
import pypolychord
from pypolychord.settings import PolyChordSettings
from pypolychord.priors import UniformPrior

try:
    from mpi4py import MPI
except ImportError:
    pass

import pandas as pd
import numpy as np
import scipy.integrate as sci


c = 299792458
 
def f1(x,m): #Flat LCDM Hubble parameter
    y = m*(1+x)**(3)+(1-m)
    return y 

def distfunc(x,params): #The function on the inside of the luminosity distance integral
    return 1/np.sqrt(f1(x,*params))

#| Load in Pantheon+ Data, and remove low redshift SNe from the relevant data
df = pd.read_table('pantheon1.txt', sep = ' ',engine='python')
cov = np.reshape(np.loadtxt('Pantheon+SH0ES_STAT+SYS.cov.txt'), [1701,1701])
mask = (df['zHD'] > 0.023) | (df['IS_CALIBRATOR'] == 1)
cephmask = (df['IS_CALIBRATOR']==0)[mask]

mbcorr = df['m_b_corr'].to_numpy()[mask]
z = df['zHD'].to_numpy()[mask]
cephdist = df['CEPH_DIST'].to_numpy()[mask]
mcov = cov[mask, :][:, mask]

mcovinv = np.linalg.inv(mcov)
mcovlogdet = np.linalg.slogdet(mcov)[1]
N = len(z)

#| Define the Likelihood
def Likelihood(theta):
    nDims = len(theta)
    Mb = theta[0]
    h = theta[1]
    params = theta[2:]  

    #| Checks for unphysical universes
    H2 = f1(z,*params)
    if any(H2<=0): #Any negative H2 expressions are unphysical, therefore have likelihood 0
      return -np.inf, []
    DC0 = sci.quad(distfunc,0,z[0],args=(params,))[0] #Distance to first SNe
    DC = np.append([0],sci.cumtrapz(1/np.sqrt(H2),z)) + DC0 #Summing distances to the remaining SNe
    comov = DC
    
    #| Adjusting for geometry
    if Curved == True:
      k = Cparams[1]
      comov = sinn(DC,k) 
      if any(comov<= 0): #This means the universe is too closed, i.e. the max distance is further than the universe's radius
        return -np.inf, []
    
    MU = 5*np.log10((c/(1000*h))*np.multiply((1+z),comov))+25 #+25 from converting to parsecs inside the log

    hr1 = mbcorr-MU  # Combining the cephied/non-cephied residuals using the cmask
    hr2 = mbcorr-cephdist
    hr = (hr1*cmask + hr2*(1-cmask))-Mb
    
    logL = float(-0.5*(np.dot(hr,np.dot(mcovinv,hr)) + mcovlogdet + N*np.log(2*np.pi)))
    return logL, []

    
#| Define a box uniform prior

def prior(hypercube):
    return priorlower+(priorupper-priorlower)*UniformPrior(0,1)(hypercube)  


#| initialise polychord settings
nDerived = 0
nDims = 3
priorlower = np.array([-20,50,0])
priorupper = np.array([-18,100,1])
    
settings = PolyChordSettings(nDims, nDerived)
settings.nlive = 1000
settings.do_clustering = True
settings.read_resume = True
settings.file_root = 'Gaussflcdm'
    
    
Output = pypolychord.run_polychord(Likelihood, nDims, nDerived, settings, prior)

#| Create a paramnames file

paramnames = [('p%i' % i, r'\theta_%i' % i) for i in range(nDims)]
paramnames += [('r*', 'r')]
output.make_paramnames_files(paramnames)

