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
h0 = 73.3
c = 299792458

#| The actual Hubble Parameter Calculation
def E(x,m,w): 
    return np.sqrt(m*(1+x)**(3) +(1-m)*(1+x)**(3*(1+6*w-3)))
    
def f(x,m,w):
    return 1/E(x,m,w)
    
def modelMU(x,theta):
    mb, h = theta[0:2]
    params = np.delete(theta, [0,1])
    y = f(x,*params)
    p0 = sci.quad(f,0,x[0],args=(*params,))[0]
    cumultrapz = np.append([0],sci.cumtrapz(y,x))
    lumdist = (c/(100000*h))*np.multiply((np.ones(len(x))+x),(p0+cumultrapz))
    return 5*np.log10(lumdist) + 25 -(2*mb+18)
    

df = pd.read_table('pantheon1.txt', sep = ' ',engine='python')
cov = np.reshape(np.loadtxt('Pantheon+SH0ES_STAT+SYS.cov.txt'), [1701,1701])

#| Where the systematics are chosen for the analysis
cutoff = 0.03 
#df.loc[(df['zHD'] < cutoff)] = 0

mask = (df['zHD'] > cutoff) & (df['IS_CALIBRATOR'] == 0)

mbcorr = df['m_b_corr'].to_numpy()[mask]
z = df['zHD'].to_numpy()[mask]
mcov = cov[mask, :][:, mask]
mcovinv = np.linalg.inv(mcov)


def LCDMlihood(theta):
    nDims = len(theta)
    hr = mbcorr-modelMU(z,theta)
    logL = float(-0.5*np.dot(hr,np.dot(mcovinv,hr)))
    return logL, []
    
#| Define a box uniform prior from -1 to 1

def prior(hypercube):
    """ Uniform prior from [-1,1]^D. """
    return UniformPrior(0,1)(hypercube)


#| Initialise the settings
nDims = 4
nDerived = 0
settings = PolyChordSettings(nDims, nDerived)
settings.file_root = 'bayescos'
settings.nlive = 400
settings.do_clustering = True
settings.read_resume = False

#| Run PolyChord

output = pypolychord.run_polychord(LCDMlihood, nDims, nDerived, settings, prior)
