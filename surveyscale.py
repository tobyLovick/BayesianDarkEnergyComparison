from numpy import pi, log, sqrt
import pypolychord
from pypolychord.settings import PolyChordSettings
from pypolychord.priors import UniformPrior
import cmath
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


def sinn(DC,k=0): #The Sinn function, which accounts for the geometry. It returns either sinx, sinhx, or x depending on the sign of the curvature
    FLAT = ((np.pi/2)**2-cmath.phase(cmath.sqrt(-1)*k)**2)/(np.pi/2)**2
    MOD = (cmath.sinh(((cmath.sqrt(k)*DC)))/(cmath.sqrt(k)+FLAT))
    return (1-FLAT)*MOD.real+FLAT*DC

df = pd.read_table('pantheon1.txt', sep = ' ',engine='python')
cov = np.reshape(np.loadtxt('Pantheon+COV.cov.txt'), [1701,1701])

Zcutoff = 0.023 #Where the systematics are chosen for the analysis
Mcutoff = 10
#df.loc[(df['zHD'] < cutoff)] = 0

mask = (df['zHD'] > Zcutoff) & (df['IS_CALIBRATOR'] == 0) #& (df['HOST_ANGSEP'] != -9.)#& (df['HOST_LOGMASS'] < Mcutoff)

mbcorr = df['m_b_corr'].to_numpy()[mask]
z = df['zHD'].to_numpy()[mask]
mcov = cov[mask, :][:, mask]
mcovinv = np.linalg.inv(mcov)
mcovlogdet = np.linalg.slogdet(mcov)[1]
N = len(z)

ID = (df['IDSURVEY'].to_numpy()[mask])
SURVEYLIST = np.unique(ID)
surveymat = np.zeros([len(ID),len(SURVEYLIST)])
surveycounter = 0
for i in SURVEYLIST:
    surveymat[:,surveycounter] = (np.isin(ID,i)*1)
    surveycounter += 1 # surveycounter doubles up as len(SURVEYLIST)

counter = 0
def LCDMlihood(theta):
    
    nDims = len(theta)
    Mb = theta[-(i[1]+2)]
    h = theta[-(i[1]+1)]
    params = theta[-i[1]:]
    
      
    scales = np.array(theta[:surveycounter])
    scalevec = np.dot(surveymat,scales**-1)
    SINV = np.diag(scalevec)
    SLOGDET = -np.log(scalevec).sum()
    
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
    
    hr = (mbcorr-Mb)-MU
    
    scaledhr = np.dot(SINV,hr)
    logL = float(-0.5*(np.dot(scaledhr,np.dot(mcovinv,scaledhr)) + mcovlogdet+ 2*SLOGDET + N*np.log(2*np.pi) ))
    return logL, []
'''    global counter
    counter += 1
    
    if counter == 1000:
      print(logL)
      counter = 0
      print(theta)
      print(Mb,h,params)
      print(SINV)
      print(scaledhr)
      print(hr)
      print('\n')'''
    

    
#| Define a box uniform prior

def prior(hypercube):
    return priorlower+(priorupper-priorlower)*UniformPrior(0,1)(hypercube)  


#Iterate over each cosmology, where funcdict stores the settings for each cosmology (filename, ndims e.t.c.)

#funcdict = [[f1,1,'flcdm',False,[0],[1]],[f2,2,'fwcdm',False,[0,-2],[1,2]],[f3,2,'slowroll',False,[0,-2],[1,1]],[f4,2,'bimetric',False,[0,0],[1,6]],[f5,3,'algthaw',False,[0,-2,-4],[1,2,4]],[f6,3,'neutrino',False,[0,0,0],[1,0.25,0.4]],[f7,2,'trans',False,[0,-0.4],[1,0.6]],[f9,2,'clcdm',True,[0,-0.5],[1,0.5]],[f10,3,'cwcdm',True,[0,-0.5,-2],[1,0.5,2]]]
funcdict = [[f1,1,'flcdm',False,[0],[1]]]
nDerived = 0
nNuisance = 2+surveycounter
for i in funcdict:
    nDims = i[1]+nNuisance
    f = i[0]
    Curved = i[3]
    priorlower = np.concatenate((0.5*np.ones(surveycounter),np.array([-20,50]),np.array(i[4])))
    priorupper = np.concatenate((2*np.ones(surveycounter),np.array([-18,100]),np.array(i[5])))
    settings = PolyChordSettings(nDims, nDerived)
    settings.nlive = 200
    settings.do_clustering = True
    settings.read_resume = True
    settings.maximise = True
    settings.file_root = 'surveyscale'+i[2]
    Output = pypolychord.run_polychord(LCDMlihood, nDims, nDerived, settings, prior)
