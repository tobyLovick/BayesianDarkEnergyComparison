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
from scipy import linalg
from scipy import special

c = 299792458


#| Hubble Parameter via different models
 
def f1(x,m): #Flat LCDM D=3
    y = m*(1+x)**(3)+(1-m)
    return y 

def f2(x,m,w): #Flat wCDM D=4
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

def f9(x,m,k): #Curved LCDM D=4
    y = m*(1+x)**(3)+k*(1+x)**(2)+(1-m-k)
    return y
        
        
def f10(x,m,k,w): #Curved wCDM D=5
    y = m*(1+x)**(3)+k*(1+x)**(2)+(1-m-k)*(1+x)**(3*(1+w))
    return y


#| Defining the likelihood functions

def gengauss(hr,a,B,K): #Generalised Gaussian For all choices of A,B,K parameters
    A=np.exp(a)
    if np.abs(K) <= 0.001:
      distance = 0.5*(np.dot(hr,np.dot(mcovinv,hr))/A)**B
      
    else:
      skewNORM = K*A**(-0.5)*np.dot(MCminushalf,hr)
      if any(skewNORM > 1):
        return -np.inf
      logsum = np.log(1-skewNORM).sum()
      distance = 0.5*(np.linalg.norm((1/K)*np.log(1-skewNORM)))**(2*B)+logsum  
    
    return float(np.log(B)+special.loggamma(N/2)-special.loggamma(N/(2*B))-0.5*(N*np.log(np.pi*A*(2**(1/B)))+mcovlogdet)-distance)


def ugengauss(hr,a,B,K):#Univariate Gaussian Generalisation, for all choices of A,B,K parameters
    A=np.exp(a)
    Bflex=1+1/(2*B)
    if np.abs(K) <= 0.001:
      NORM = (A**(-0.5))*np.dot(MCminushalf,hr)
      distance = 0.5*((np.abs(NORM)**(2*B)).sum())
    else:
      skewNORM = K*A**(-0.5)*np.dot(MCminushalf,hr)
      if any(skewNORM > 1):
        return -np.inf
      logsum = np.log(1-skewNORM).sum()
      distance = 0.5*((np.abs((1/K)*np.log(1-skewNORM))**(2*B)).sum())+logsum 
    return float(-N*(0.5*np.log(A)+special.loggamma(Bflex)+Bflex*np.log(2))-0.5*mcovlogdet-distance)
  
    
    
    

def L1(hr,params): #Gaussian
    return float(-0.5*(np.dot(hr,np.dot(mcovinv,hr)) + mcovlogdet + N*np.log(2*np.pi)))
    
def L2(hr,params):#A
    return gengauss(hr,params[0],1,0)
    
def L3(hr,params):#B
    return gengauss(hr,0,params[0],0)
def L4(hr,params):#K
    return gengauss(hr,0,1,params[0])   

def L5(hr,params):#AB
    return gengauss(hr,params[0],params[1],0)

def L6(hr,params):#AK
    return gengauss(hr,params[0],1,params[1])
    
def L7(hr,params):#BK
    return gengauss(hr,0,params[0],params[1])
    
def L8(hr,params):#ABK
    return gengauss(hr,params[0],params[1],params[2])
        
def L9(hr,params):#Student's t
    NU = 10**params[0]
    sigmalogdet = mcovlogdet + (N*np.log((NU-2)/NU))
    power = 1 + np.dot(hr,np.dot(mcovinv,hr))/((NU-2))  
    return float(-0.5*N*np.log(NU*np.pi) + special.loggamma((NU+N)/2)-special.loggamma(NU/2)-0.5*sigmalogdet+(-0.5*(NU+N)*np.log(power)))

def L10(hr,params):#UB  
    return ugengauss(hr,0,params[0],0) 
    
def L11(hr,params):#UAB
    return ugengauss(hr,params[0],params[1],0)
    
def L12(hr,params):#UBK
    return ugengauss(hr,0,params[0],params[1]) 
    
def L13(hr,params):#UABK
    return ugengauss(hr,params[0],params[1],params[2])
    

def stu(x,v): #1D student's t, seperated out from L14 for readability
  return (special.loggamma((v+1)/2)-special.loggamma(v/2)-0.5*np.log(np.pi*v)-0.5*(v+1)*(np.log(1 +(x**2)/((v-2)))))


def L14(hr,params):#Ustu
    NU = 10**params[0]
    sigmalogdet = mcovlogdet + np.log((NU-2)/NU)
    NORM = np.dot(MCminushalf,hr)
    return float(-0.5*sigmalogdet+stu(NORM,NU).sum())
    
    

def distfunc(x,params): #The function on the inside of the luminosity distance integral
    return 1/np.sqrt(f(x,*params))


def sinn(DC,k=0): #The Sinn function, which accounts for the geometry
    if k>0: 
        return np.sinh(DC*np.sqrt(k))/np.sqrt(k)
    elif k == 0:
        return DC
    else:
        return np.sin(DC*np.sqrt(-k))/np.sqrt(-k)

#| Load in Pantheon+ Data, and remove low redshift SNe from the relevant data
df = pd.read_table('pantheon1.txt', sep = ' ',engine='python')
cov = np.reshape(np.loadtxt('Pantheon+SH0ES_STAT+SYS.cov.txt'), [1701,1701])
vpecsystmatics = np.reshape(np.loadtxt('Pantheon+SH0ES_122221_VPEC.cov.txt'), [1701,1701])

vpeccov = cov-vpecsystematics

mask = (df['zHD'] > 0.023) | (df['IS_CALIBRATOR'] == 1)
mbcorr = df['m_b_corr'].to_numpy()[mask]
zHD = df['zHD'].to_numpy()[mask]
zCMB = df['zCMB'].to_numpy()[mask]
bias=df['biasCor_m_b'].to_numpy()[mask]
mcov = cov[mask, :][:, mask]
mvpeccov = vpeccov[mask, :][:, mask]

mcovinv = np.linalg.inv(mcov)
mvpeccovinv = np.linalg.inv(mvpeccov)

MCminushalf = linalg.sqrtm(mcovinv)

mcovlogdet = np.linalg.slogdet(mcov)[1]
cephdist = df['CEPH_DIST'].to_numpy()[mask]
cmask = (df['IS_CALIBRATOR']==0)[mask]

N = len(zHD)

#| Define the Likelihood
def Likelihood(theta):
    nDims = len(theta)
    Sparams = theta[:Smodel[1]]
    MB = theta[Smodel[1]]
    h = theta[Smodel[1]+1]
    Cparams = theta[-Cmodel[1]:]  

    #| Checks for unphysical universes
    H2 = f(z,*Cparams)
    if any(H2<=0): #Any negative H2 expressions are unphysical, therefore have likelihood 0
      return -np.inf, []
    DC0 = sci.quad(distfunc,0,z[0],args=(Cparams,))[0] #Distance to first SNe
    DC = np.append([0],sci.cumtrapz(1/np.sqrt(H2),z)) + DC0 #Summing distances to the remaining SNe
    comov = DC
    
    #| Adjusting for geometry
    if Curved == True:
      k = Cparams[1]
      comov = sinn(DC,k) 
      if any(comov<= 0): #This means the universe is too closed, i.e. the max distance is further than the universe's radius
        return -np.inf, []
    
    MU = 5*np.log10((c/(1000*h))*np.multiply((1+z),comov))+25 #+25 from converting to parsecs inside the log

    hr1 = mb-MU  # Combining the cephied/non-cephied residuals using the cmask
    hr2 = mb-cephdist
    hr = (hr1*cmask + hr2*(1-cmask))-MB
    
    logL = Smodel[0](hr,Sparams)
    
    return logL, []

    
#| Define a box uniform prior

def prior(hypercube):
    return priorlower+(priorupper-priorlower)*UniformPrior(0,1)(hypercube)  


#| Set up the settings for each cosmology/scatter model

Aprior=[-np.log(5),np.log(5)] ##Comment out as appropriate for log/uniform prior
#Aprior=[0.2,5]


cosdict = [[f1,1,'flcdm',False,[0],[1]],[f2,2,'fwcdm',False,[0,-2],[1,2]],[f3,2,'slowroll',False,[0,-2],[1,1]],[f4,2,'bimetric',False,[0,0],[1,6]],[f5,3,'algthaw',False,[0,-2,-4],[1,2,4]],[f6,3,'neutrino',False,[0,0,0],[1,0.25,0.4]],[f7,2,'trans',False,[0,-0.4],[1,0.6]],[f9,2,'clcdm',True,[0,-0.5],[1,0.5]],[f10,3,'cwcdm',True,[0,-0.5,-2],[1,0.5,2]]]

likedict = [[L1,0,'Gauss',[],[]],[L2,1,'A',[Aprior[0]],[Aprior[1]]],[L3,1,'B',[0.01],[3]],[L4,1,'K',[-0.2],[0.2]],[L5,2,'AB',[Aprior[0],0.01],[Aprior[1],3]],[L6,2,'AK',[Aprior[0],-0.2],[Aprior[1],0.2]],[L7,2,'BK',[0.01,-0.2],[3,0.2]],[L8,3,'ABK',[Aprior[0],0.01,-0.2],[Aprior[1],3,0.2]],[L9,1,'Stu',[0.3011],[5]],[L10,1,'UB',[0.01],[3]],[L11,2,'UAB',[Aprior[0],0.01],[Aprior[1],3]],[L12,2,'UBK',[0.01,-0.2],[3,0.2]],[L13,3,'UABK',[Aprior[0],0.01,-0.2],[Aprior[1],3,0.2]],[L14,1,'UStu',[0.3011],[5]]]


#| Import model choice from cmdline args
arg = int(sys.argv[1])

Biastest,model=divmod(arg,126)
Snumber,Cnumber = divmod(model,9) 
Smodel = likedict[Snumber]
Cmodel = cosdict[Cnumber]

if Biastest==1:
    z=zCMB
    mcovinv = mvpeccovinv # Removes the systematic error from peculiar velocity corrections from the covariance matrix
else:
    z=zHD

if Biastest==2:
    mb=mbcorr+bias
else:
    mb=mbcorr

#| initialise polychord settings
nDerived = 0
nDims = Cmodel[1]+Smodel[1]+2
f = Cmodel[0]
Curved = Cmodel[3]
priorlower = np.concatenate((np.array(Smodel[3]),np.concatenate((np.array([-20,50]),np.array(Cmodel[4])))))
priorupper = np.concatenate((np.array(Smodel[4]),np.concatenate((np.array([-18,100]),np.array(Cmodel[5])))))
    
settings = PolyChordSettings(nDims, nDerived)
settings.nlive = 10000
settings.do_clustering = True
settings.read_resume = True
settings.file_root = ["","VpecUncorrected","BiasUncorrected"][Biastest]+Smodel[2]+Cmodel[2]
    
    
Output = pypolychord.run_polychord(Likelihood, nDims, nDerived, settings, prior)
