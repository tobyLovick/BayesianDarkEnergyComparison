from numpy import pi, log, sqrt
import pypolychord
from pypolychord.settings import PolyChordSettings
from pypolychord.priors import UniformPrior

try:
    from mpi4py import MPI
except ImportError:
    pass

import sys
import pandas as pd
import numpy as np
import scipy.integrate as sci

c = 299792458


#| Hubble Parameter via different models
 
def f1(x,m): #Flat LCDM D=3
    y = m*(1+x)**(3)+(1-m)
    return y 

def f2(x,m,w): #wCDM D=4
    y =m*(1+x)**(3)+(1-m)*(1+x)**(3*(1+w))
    return y
    
def f3(x,m,d): #Slow-Roll D=4
    y = m*(1+x)**(3)+(1-m)*((1+x)**3/(m*(1+x)**3+1-m))**(d/(1-m))
    return y

def f4(x,m,B1): #Bimetric Gravity D=4
    B0 = 3*(1-m)-(B1)**2
    G = 0.5*m*(1+x)**(3) + B0/6
    y = G + np.sqrt(G**2+((B1)**2)/3)
    return y

def f5(x,m,w,p): #Algebraic Thawing D=5
    alpha = 1/(1.3)
    y = m*(1+x)**3+(1-m)*np.exp(((1+w)/(alpha*p))*(1-(1-alpha + alpha/((1+x)**3))**p))
    return y

def f6(x,m,e,v): #Growing Neutrino Mass D=5
    A = 1/(1+x)
    DS = np.array(((1-m)*A**3+2*v*(A**1.5-A**3))/(1-(1-m)*(1-A**3)+2*v*(A**1.5-A**3)))
    DS[DS < e] = e
    y = (m*(1+x)**3)/(1-DS)
    return y

def f7(x,m,d): #Dark Energy Tranisiton D=4
    S = 0.5*(1-np.tanh((x-0.1)/(0.01)))
    S0 = 0.5*(1-np.tanh  (-10))
    y = m*(1+x)**3+(1-m)*(1+(2*d*S)/((1+m)*S0))
    return y

def f9(x,m,k): #CURVEDLCDM D=4
    y = m*(1+x)**(3)+k*(1+x)**(2)+(1-m-k)
    return y
        
        
def f10(x,m,k,w): #CURVEDwCDM D=5
    y = m*(1+x)**(3)+k*(1+x)**(2)+(1-m-k)*(1+x)**(3*(1+w))
    return y

def distfunc(x,params): #The function on the inside of the luminosity distance integral
    return 1/np.sqrt(f(x,*params))


def sinn(DC,k=0): #The Sinn function, which accounts for the geometry
    if k>0: 
        return np.sinh(DC*np.sqrt(k))/np.sqrt(k)
    elif k == 0:
        return DC
    else:
        return np.sin(DC*np.sqrt(-k))/np.sqrt(-k)

df = pd.read_table('pantheon1.txt', sep = ' ',engine='python')
cov = np.reshape(np.loadtxt('Pantheon+SH0ES_STAT+SYS.cov.txt'), [1701,1701])

Zcutoff = 0.023 #Where the systematics are chosen for the analysis

mask = (df['zHD'] > Zcutoff) | (df['IS_CALIBRATOR'] == 1)

mbcorr = df['m_b_corr'].to_numpy()[mask]
z = df['zHD'].to_numpy()[mask]
zvpec = df['zCMB'].to_numpy()[mask]
mcov = cov[mask, :][:, mask]
mcovinv = np.linalg.inv(mcov)
mcovlogdet = np.linalg.slogdet(mcov)[1]

cephdist = df['CEPH_DIST'].to_numpy()[mask]
cmask = (df['IS_CALIBRATOR']==0)[mask] # This mask allows us to use the cephied calibrator distances for our hubble residuals
MBmask = (df['zHD'] > 0.1 )[mask]

N = len(z)
zmax = z.max()

#| Vpec likelihood (uses zvpec)
def likelihoodvpec(theta):
    nDims = len(theta)
    Mb = theta[0]
    h = theta[1]
    params = theta[-model[1]:]  
    #| Checks for unphysical universes
    H2 = f(zvpec,*params)
    if any(H2<=0): #Any negative H2 expressions are unphysical, therefore have likelihood 0
      return -np.inf, []
    DC0 = sci.quad(distfunc,0,zvpec[0],args=(params,))[0] #Distance to first SNe
    DC = np.append([0],sci.cumtrapz(1/np.sqrt(H2),zvpec)) + DC0 #Summing distances to the remaining SNe
    comov = DC
    
    if Curved == True:
      k = params[1]
      comov = sinn(DC,k) #Adjusting for geometry
      if any(comov<= 0): #This means the universe is too closed, i.e. the max distance is further than the universe's radius
        return -np.inf, []
    
    MU = 5*np.log10((c/(1000*h))*np.multiply((1+zvpec),comov))+25 #+25 from converting to parsecs inside the log

    hr1 = mbcorr-MU  # Combining the cephied/non-cephied residuals using the cmask
    hr2 = mbcorr-cephdist
    hr = (hr1*cmask + hr2*(1-cmask))-Mb
    
    logL = float(-0.5*(np.dot(hr,np.dot(mcovinv,hr)) + mcovlogdet + N*np.log(2*np.pi)))
    
    return logL, []

#| Double MB likelihood
def likelihooddouble(theta):
    nDims = len(theta)
    Mb1 = theta[0]
    Mb2 = theta[1]
    h = theta[2]
    params = theta[-model[1]:]  

    #| Checks for unphysical universes
    H2 = f(z,*params)
    if any(H2<=0): #Any negative H2 expressions are unphysical, therefore have likelihood 0
      return -np.inf, []
    DC0 = sci.quad(distfunc,0,z[0],args=(params,))[0] #Distance to first SNe
    DC = np.append([0],sci.cumtrapz(1/np.sqrt(H2),z)) + DC0 #Summing distances to the remaining SNe
    comov = DC
    
    if Curved == True:
      k = params[1]
      comov = sinn(DC,k) #Adjusting for geometry
      if any(comov<= 0): #This means the universe is too closed, i.e. the max distance is further than the universe's radius
        return -np.inf, []
    
    MU = 5*np.log10((c/(1000*h))*np.multiply((1+z),comov))+25 #+25 from converting to parsecs inside the log

    hr1 = mbcorr-MU  # Combining the cephied/non-cephied residuals using the cmask
    hr2 = mbcorr-cephdist
    Mb = Mb1*(1-MBmask)+Mb2*MBmask #Near and Far Absolute Magnitude
    hr = (hr1*cmask + hr2*(1-cmask))-Mb
    
    logL = float(-0.5*(np.dot(hr,np.dot(mcovinv,hr)) + mcovlogdet + N*np.log(2*np.pi)))
    
    return logL, []
    
#| Linear MB likelihood

def likelihoodlinear(theta):
    nDims = len(theta)
    Mb1 = theta[0]
    Mb2 = theta[1]
    h = theta[2]
    params = theta[-model[1]:]  

    #| Checks for unphysical universes
    H2 = f(z,*params)
    if any(H2<=0): #Any negative H2 expressions are unphysical, therefore have likelihood 0
      return -np.inf, []
    DC0 = sci.quad(distfunc,0,z[0],args=(params,))[0] #Distance to first SNe
    DC = np.append([0],sci.cumtrapz(1/np.sqrt(H2),z)) + DC0 #Summing distances to the remaining SNe
    comov = DC
    
    if Curved == True:
      k = params[1]
      comov = sinn(DC,k) #Adjusting for geometry
      if any(comov<= 0): #This means the universe is too closed, i.e. the max distance is further than the universe's radius
        return -np.inf, []
    
    MU = 5*np.log10((c/(1000*h))*np.multiply((1+z),comov))+25 #+25 from converting to parsecs inside the log

    hr1 = mbcorr-MU  # Combining the cephied/non-cephied residuals using the cmask
    hr2 = mbcorr-cephdist
    Mb = Mb1 + (Mb2-Mb1)*(z/zmax)#Mb with a linear gradient
    hr = (hr1*cmask + hr2*(1-cmask))-Mb
    
    logL = float(-0.5*(np.dot(hr,np.dot(mcovinv,hr)) + mcovlogdet + N*np.log(2*np.pi)))
    
    return logL, []


    
#| Define a box uniform prior

def prior(hypercube):
    return priorlower+(priorupper-priorlower)*UniformPrior(0,1)(hypercube)  


#Iterate over each cosmology, where funcdict stores the settings for each cosmology (filename, ndims e.t.c.)

funcdict = [[f1,1,'flcdm',False,[0],[1]],[f2,2,'fwcdm',False,[0,-2],[1,2]],[f3,2,'slowroll',False,[0,-2],[1,1]],[f4,2,'bimetric',False,[0,0],[1,6]],[f5,3,'algthaw',False,[0,-2,-4],[1,2,4]],[f6,3,'neutrino',False,[0,0,0],[1,0.25,0.4]],[f7,2,'trans',False,[0,-0.4],[1,0.6]],[f9,2,'clcdm',True,[0,-0.5],[1,0.5]],[f10,3,'cwcdm',True,[0,-0.5,-2],[1,0.5,2]]]

#| Import model choice from cmdline args
modelnumber = int(sys.argv[1])

model = funcdict[0]
nDims = 4

if modelnumber<9:
  model = funcdict[modelnumber]
  nDims = model[1]+2
  fileroot = 'Uncorrected'+model[2]
  Likelihood = likelihoodvpec
  priorlower = np.concatenate((np.array([-20,50]),np.array(model[4])))
  priorupper = np.concatenate((np.array([-18,100]),np.array(model[5])))
elif modelnumber==9:
  Likelihood = likelihooddouble
  fileroot = 'doubleMB'
  priorlower = np.array([-20,-20,50,0])
  priorupper = np.array([-18,-18,100,1])  
else:
  Likelihood = likelihoodlinear
  fileroot = 'linearMB'
  priorlower = np.array([-20,-20,50,0])
  priorupper = np.array([-18,-18,100,1])

print(fileroot)
#| initialise polychord settings
nDerived = 0
f = model[0]
Curved = model[3]
settings = PolyChordSettings(nDims, nDerived)
settings.nlive = 10000
settings.do_clustering = True
settings.read_resume = True
settings.file_root = fileroot
Output = pypolychord.run_polychord(Likelihood, nDims, nDerived, settings, prior)

