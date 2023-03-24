from numpy import pi, log, sqrt
import pypolychord
from pypolychord.settings import PolyChordSettings
from pypolychord.priors import UniformPrior
try:
    from mpi4py import MPI
except ImportError:
    pass
from astropy.cosmology import wCDM

import pandas as pd
import numpy as np
import scipy.integrate as sci
h0 = 73.3
c = 299792458


#| Hubble Parameter via different models
 
def f1(x,m): #Flat LCDM D=2
    E = np.sqrt(m*(1+x)**(3)+(1-m)) 
    return 1/E 

def f2(x,m,w): #wCDM D=3
    W = 6*w-3
    E = np.sqrt(m*(1+x)**(3)+(1-m)*(1+x)**(3*(1+W))) 
    return 1/E

def modelMU(theta):
    mb, h = theta[0:2]
    params = theta[2:]
    y = f2(z,*params)
    p0 = sci.quad(f2,0,z[0],args=(*params,))[0]
    cumultrapz = np.append([0],sci.cumtrapz(y,z))
    lumdist = (c/(100000*(0.5*h+0.5)))*np.multiply((1+z),(p0+cumultrapz))
    return 5*np.log10(lumdist)+25

df = pd.read_table('pantheon1.txt', sep = ' ',engine='python')
print('DF loaded')
cov = np.reshape(np.loadtxt('Pantheon+SH0ES_STAT+SYS.cov.txt'), [1701,1701])
print('Matrix Loaded')

Zcutoff = 0.02 #Where the systematics are chosen for the analysis
Mcutoff = 10

mask = (df['zCMB'] > Zcutoff) or (df['IS_CALIBRATOR'] == 1)

mbcorr = df['m_b_corr'].to_numpy()[mask]
z = df['zHD'].to_numpy()[mask]
cephdist = df['CEPH_DIST'].to_numpy()[mask]

mcov = cov[mask, :][:, mask]
mcovinv = np.linalg.inv(mcov)

calibratormask = np.array((df['IS_CALIBRATOR']==0)[mask])

    
def Likelihood(theta):
    nDims = len(theta)
    MB = 2*theta[0]+18
    hr1 = mbcorr-modelMU(theta)-MB
    hr2 = mbcorr-cephdist-MB
    hr = np.multiply(calibratormask,hr1) + np.multiply(np.ones(len(z))-calibratormask,hr2)
    logL = float(-0.5*np.dot(hr,np.dot(mcovinv,hr)))
    return logL, []    
    
#| Define a box uniform prior from -1 to 1

def prior(hypercube):
  
    return UniformPrior(0,1)(hypercube)


#| Initialise the settings

nDims = 3
nDerived = 0
settings = PolyChordSettings(nDims, nDerived)
settings.file_root = 'CephFLCDM'
settings.nlive = 500
settings.do_clustering = True
settings.read_resume = True

#| Run PolyChord

output = pypolychord.run_polychord(Likelihood, nDims, nDerived, settings, prior)

