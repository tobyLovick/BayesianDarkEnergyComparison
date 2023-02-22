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

def f0(x,m): #EdS D=2
    M = (0*m)+1
    E = np.sqrt(M*(1+x)**(3)) 
    return 1/E
      
def f1(x,m): #Flat LCDM D=3
    E = np.sqrt(m*(1+x)**(3)+(1-m)) 
    return 1/E 

def f2(x,m,w): #wCDM D=4
    W = 6*w-3
    E = np.sqrt(m*(1+x)**(3)+(1-m)*(1+x)**(3*(1+W))) 
    return 1/E
    
def f2R(x,m,w): #wCDM w/ radiation D=4
    W = 6*w-3
    E = np.sqrt(m*(1+x)**(3)+0.00008*(1+x)**4+(1-m-0.00008)*(1+x)**(3*(1+W))) 
    return 1/E
    
def f3(x,m,d): #Slow-Roll D=4
    D = 3*d-2
    E = np.sqrt(m*(1+x)**(3)+(1-m)*((1+x)**3/(m*(1+x)**3+1-m))**(D/(1-m)))
    return 1/E

def f4(x,m,B): #Bimetric Gravity D=4
    B1 = B*6
    B0 = 3*(1-m)-(B1)**2
    G = 0.5*m*(1+x)**(3) + B0/6
    E = np.sqrt(G + np.sqrt(G**2+((B1)**2)/3))
    return 1/E

def f5(x,m,w,p): #Algebraic Thawing D=5
    alpha = 1/(1.3)
    P = (8*p-4)/3
    W = 8*w-4
    E = np.sqrt(m*(1+x)**3+(1-m)*np.exp(((1+W)/(alpha*P))*(1-(1-alpha + alpha/((1+x)**3))**P)))
    return 1/E

def f6(x,m,e,v): #Growing Neutrino Mass D=5
    V = 0.4*v
    A = 1/(1+x)
    DS = np.array(((1-m)*A**3+2*V*(A**1.5-A**3))/(1-(1-m)*(1-A**3)+2*V*(A**1.5-A**3)))
    DS[DS < 0.25*e] = 0.25*e
    E = np.sqrt((m*(1+x)**3)/(1-DS))
    return 1/E

def f7(x,m,D): #Dark Energy Tranisiton D=4
    d = D-0.4
    S = 0.5*(1-np.tanh((x-0.1)/(0.01)))
    S0 = 0.5*(1-np.tanh(-10))
    E = np.sqrt(m*(1+x)**3+(1-m)*(1+(2*d*S)/((1+m)*S0)))
    return 1/E

def f8(x,q,j,s): #Cosmographic Expansion D=5
    Q = 10*q-5
    J = 10*j-5
    S = 20*s-10
    E = 1+(1+Q)*x+0.5*(J-Q**2)*x**2+(1/6)*(3*Q**3+3*Q**2+J*(3+4*Q)-S)*x**3
    return 1/E
    
def f9(x,M,K): #CURVEDLCDM
    m = 0.88*M + 0.12
    k = K - 0.5
    E = np.sqrt(m*(1+x)**(3)+k*(1+x)**(2)+(1-m-k)) 
    return 1/E 
  
def f10(x,M,K,W): #curvedwCDM
    w = 6*W-3
    m = 0.88*M + 0.12
    k = K - 0.5
    E = np.sqrt(m*(1+x)**(3)+k*(1+x)**(2)+(1-m-k)*(1+x)**(3*(1+w)))  
    return 1/E

def modelMU(theta):
    mb, h = theta[0:2]
    params = theta[2:]
    y = f1(z,*params)
    p0 = sci.quad(f1,0,z[0],args=(*params,))[0]
    cumultrapz = np.append([0],sci.cumtrapz(y,z))
    lumdist = (c/(100000*(0.5*h+0.5)))*np.multiply((1+z),(p0+cumultrapz))
    return 5*np.log10(lumdist)+25-(2*mb+18)

'''def curvedmodelMU(theta): #A different method using astropy for calculating curved cosmologies, OwCDM and LCDM
    mb, h = theta[0:2]
    params = theta[2:]
    #k = -1*(theta[3]-0.5)*((0.5*h+0.5)/(3.086*10**17))**2/(c**2)
    k = theta[3]-0.5
    y = f9(z,*params)
    p0 = sci.quad(f9,0,z[0],args=(*params,))[0]
    cumultrapz = p0+np.append([0],sci.cumtrapz(y,z))
    if k>0:
        comovdist = np.sinh((cumultrapz)*np.sqrt(k))/np.sqrt(k)
    elif k == 0:
        comovdist = (cumultrapz)
    else:
        comovdist = np.sin((cumultrapz)*np.sqrt(-k))/np.sqrt(-k)
    lumdist = (c/(100000*(0.5*h+0.5)))*np.multiply((1+z),comovdist)
    return 5*np.log10(lumdist)+25-(2*mb+18)'''

def curvedmodelMU(theta):
    mb, h = theta[0:2]
    params = theta[2:]
    #k = -1*(theta[3])*(h/(3.086*10**19))**2/(c**2)
    k = theta[3]-0.5
    y = f10(z,*params)
    p0 = sci.quad(f10,0,z[0],args=(*params,))[0]
    cumultrapz = np.append([0],sci.cumtrapz(y,z))
    #print(k)
    if k>0:
        comovdist = np.sinh((p0+cumultrapz)*np.sqrt(k))/np.sqrt(k)
    elif k == 0:
        comovdist = (p0+cumultrapz)
    else:
        comovdist = np.sin((p0+cumultrapz)*np.sqrt(-k))/np.sqrt(-k)
    #comovdist = (p0+cumultrapz)
    lumdist = (c/(100000*(0.5*h+0.5)))*np.multiply((1+z),comovdist)
    return 5*np.log10(lumdist)+25-(2*mb+18)  

'''def curvedmodelMU(theta): #A different method using astropy for calculating curved cosmologies, OwCDM and LCDM
    mb, h ,m , k, w= theta
    k += -0.5
    m = 0.88*m+0.12
    H0 = h*50+50
    w = 6*w-3
    cosmo = wCDM(H0 = H0, Om0 = m, Ode0 = 1-m-k, w0 = w) 
    lumdist = cosmo.luminosity_distance(z).value
    return 5*np.log10(lumdist)+25-(2*mb+18)'''

def modelMUEdS(theta):
    mb, h = theta
    lumdist = (c/(100000*(0.5*h+0.5)))*EdSDL
    return 5*np.log10(lumdist)+25-(2*mb+18)

df = pd.read_table('pantheon1.txt', sep = ' ',engine='python')
print('DF loaded')
cov = np.reshape(np.loadtxt('Pantheon+SH0ES_STAT+SYS.cov.txt'), [1701,1701])
print('Matrix Loaded')

Zcutoff = 0.03 #Where the systematics are chosen for the analysis
Mcutoff = 10
#df.loc[(df['zHD'] < cutoff)] = 0

mask = (df['zHD'] > Zcutoff) & (df['IS_CALIBRATOR'] == 0) #& (df['HOST_LOGMASS'] < Mcutoff)

mbcorr = df['m_b_corr'].to_numpy()[mask]
z = df['zHD'].to_numpy()[mask]
mcov = cov[mask, :][:, mask]
mcovinv = np.linalg.inv(mcov)

'''EdSDL = 2*(1+z-(1+z)**(-0.5))'''

def LCDMlihood(theta):
    nDims = len(theta)
    hr = mbcorr-curvedmodelMU(theta)
    logL = float(-0.5*np.dot(hr,np.dot(mcovinv,hr)))
    return logL, []
    
#| Define a box uniform prior from -1 to 1

def prior(hypercube):
    """ Uniform prior from [-1,1]^D. """
    return UniformPrior(0,1)(hypercube)


#| Initialise the settings
nDims = 5
nDerived = 0
settings = PolyChordSettings(nDims, nDerived)
settings.file_root = 'CurvedwCDM'
settings.nlive = 600
settings.do_clustering = True
settings.read_resume = False

#| Run PolyChord

output = pypolychord.run_polychord(LCDMlihood, nDims, nDerived, settings, prior)

#attempt to make a corner plot

'''import corner
paramnames = [('p%i' % i, r'\theta_%i' % i) for i in range(nDims)]
paramnames += [('r*', 'r')]
df = pd.read_table('chains/AlgThaw_equal_weights.txt', sep = '  ',engine='python', names = ['a','b']+paramnames)
samples = np.zeros([nDims,len(df['a'])])
for i in range(nDims):
    samples[i,:] = df[paramnames[i]].to_numpy()
samples[0] = 2*samples[0] +18
samples[1] = samples[1]*50+50
samples[3] = 8*samples[3]-4
samples[4] = 8*samples[4]-4
figure = corner.corner(samples.T,labels = [r"$M_b$",r"$H_0$",r"$\Omega _m$", r"$w_0$",r"$p$"],show_titles=True,
    title_kwargs={"fontsize": 12})
figure.savefig('ALgThaw.pdf', format = 'pdf')'''

