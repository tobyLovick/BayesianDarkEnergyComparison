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

def E(x,m): 
    return np.sqrt(m*(1+x)**(3) +(1-m))
def f(x,m):
    return 1/E(x,m)
    
def lumdist(x,m,h):
    y = f(x,m)
    p0 = sci.quad(f,0,x[0],args=(m,))[0]
    cumultrapz = np.append([0],sci.cumtrapz(y,x))
    return (c/(100000*h))*np.multiply((np.ones(len(x))+x),(p0*np.ones(len(x))+cumultrapz))

def lumdistance(x,m,h): #gives in MPC
    if x != 0:
        return (c/(100000*h))*(1+x)*sci.quad(f,0,x,args=(m,))[0] #(c/(1000*h0))* #for correct answer
    else:
        return 10 **(-5)
def modelMU(x,m,h):
    return (5*np.log10(lumdist(x,m,h))) + 25 
    

df = pd.read_table('pantheon1.txt', sep = ' ',engine='python')
cov = np.reshape(np.loadtxt('Pantheon+SH0ES_STAT+SYS.cov.txt'), [1701,1701])
print("Matrix Loaded")

cutoff = 0.03
#df.loc[(df['zHD'] < cutoff)] = 0

mask = (df['zHD'] > cutoff) & (df['IS_CALIBRATOR'] == 0)

MU = df['MU_SH0ES'].to_numpy()[mask]
mbcorr = df['m_b_corr'].to_numpy()[mask]
z = df['zHD'].to_numpy()[mask]
mcov = cov[mask, :][:, mask]
mcovinv = np.linalg.inv(mcov)
print("Matrix Inverted")
# ALL OF THIS WORKS SO FAR!!!

def LCDMlihood(theta):
    nDims = len(theta)
    m, h= theta
    hr = MU-modelMU(z,m,h)
    logL = float(-0.5*np.dot(hr,np.dot(mcovinv,hr)))
    return logL, []
    
 ## DO this with solve instead nowW!
def likelihood(theta):
    """ Simple Gaussian Likelihood"""
    nDims = len(theta)  
    r2 = sum(theta**2)
    logL = -log(2*pi*sigma*sigma)*nDims/2.0
    logL += -r2/2/sigma/sigma

    return logL, [r2]

#| Define a box uniform prior from 0 to 1

def prior(hypercube):
    """ Uniform prior from [-1,1]^D. """
    return UniformPrior(0,1)(hypercube)


#| Initialise the settings
nDims = 2
nDerived = 0
settings = PolyChordSettings(nDims, nDerived)
settings.file_root = 'nondegenlambdacdm'
settings.nlive = 2000
settings.do_clustering = True
settings.read_resume = False

#| Run PolyChord

output = pypolychord.run_polychord(LCDMlihood, nDims, nDerived, settings, prior)

#| Create a paramnames file

paramnames = [('p%i' % i, r'\theta_%i' % i) for i in range(nDims)]
paramnames += [('r*', 'r')]
output.make_paramnames_files(paramnames)

''' attempt to make a corner plot'''
import corner
df = pd.read_table('chains/nondegenlambdacdm_equal_weights.txt', sep = '  ',engine='python', names = ['a','b']+paramnames)
samples = np.zeros([nDims,len(df['a'])])
for i in range(nDims):
    samples[i,:] = df[paramnames[i]].to_numpy()
figure = corner.corner(samples.T)
figure.savefig('LCDMpost.pdf')


#| Make an anesthetic plot (could also use getdist)
#try:
#    from anesthetic import NestedSamples
#    samples = NestedSamples(root= settings.base_dir + '/' + settings.file_root)
#    fig, axes = samples.plot_2d(['p0','p1','p2','p3','r'])
#    fig.savefig('posterior.pdf')
#
#except ImportError:
#    try:
#        import getdist.plots
#        posterior = output.posterior
#        g = getdist.plots.getSubplotPlotter()
#        g.triangle_plot(posterior, filled=True)
#        g.export('posterior.pdf')
#    except ImportError:
#        print("Install matplotlib and getdist for plotting examples")
#
#    print("Install anesthetic or getdist  for for plotting examples")
