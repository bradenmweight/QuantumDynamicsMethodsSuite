import numpy as np
import random


class parameters():
    NCPUS = 150
    dtI = 0.01
    NSteps = int(20/dtI) # Spin boson models go to 20 a.u.
    NTraj = 10000
    EStep = 50
    dtE = dtI/EStep
    NSkip = 0.1 # Plot every {} a.u.

    M = 1 # Set to "1" for spin-boson
    NStates = 2
    initState = 0
    method = "spin-PLDM" # "SQC" or "PLDM", "spin-LSC", or "spin-PLDM"
    sampling = "focused"
    dirName = "TRAJ_spin-PLDM__Model_F_10000/"


    ######## Only for SQC, unused else
    windowtype = "n-triangle" # "Square", "N-Triangle"
    adjustedgamma = "yes" # "yes", "no"
    ########

    fs_to_au = 41.341 # a.u./fs



    # SPIN BOSON MODEL PARAMETERS
    # Taking from Mannouch/Richardson spin-PLDM I (2020)

    Delta = 1.0 # Non-varied parameter. Kinda dumb.
    wm = 4.0 # Cutoff for ohmic function, personal (i.e., Frank's) choice -- 3 * wc, characteristic frequency or peak of spectral density
   

    #Model (a) -- Sym. High T
    Eps,xi,beta,wc,ndof = 0.0, 0.09, 0.1, 2.5, 100
    #Model (b) -- Sym. Low T
    #Eps,xi,beta,wc,ndof = 0.0, 0.09, 5.0, 2.5, 100
    #Model (c) -- Asym. High T
    #Eps,xi,beta,wc,ndof = 1.0, 0.1, 0.25, 1.0, 100
    #Model (d) -- Asym. Low T
    #Eps,xi,beta,wc,ndof = 1.0, 0.1, 5.0, 2.5, 100
    #Model (e) -- Sym. Strong Bath
    #Eps,xi,beta,wc,ndof = 0.0, 2.0, 1.0, 1.0, 400
    #Model (f) -- Asym. Strong Bath
    #Eps,xi,beta,wc,ndof = 5.0, 4.0, 0.1, 2.0, 400

    w0 = wc*( 1-np.exp(-wm) ) / ndof # Peak of distribution

def initModelParams():
    xi = parameters.xi
    ndof = parameters.ndof
    wc = parameters.wc
    w0 = parameters.w0
    c = np.zeros(( ndof ))
    w = np.zeros(( ndof ))
    for d in range(ndof):
        w[d] = -wc * np.log(1 - (d+1)*w0/wc)
        c[d]   = np.sqrt(xi*w0) * w[d] # V(x^2) + c*x 
    return c,w


def Hel(R):

    Eps = parameters.Eps
    Delta = parameters.Delta
    c,w = initModelParams()

    Hel = np.zeros((2,2))


    # Harmonic part is state-independent and common
    Hel[0,0] = Eps
    Hel[0,1] = Delta
    Hel[1,0] = Hel[0,1]
    Hel[1,1] = -Eps

        
    Hel[0,0]  +=  np.sum( c * R )
    Hel[1,1]  -=  np.sum( c * R )

    return Hel
  

def dHel0(R):

    c,w = initModelParams()

    return w**2 * R

def dHel(R):
      
    dHel = np.zeros(( 2,2,len(R) ))

    c,w = initModelParams()

    dHel[0,0,:] = c
    dHel[1,1,:] = -c

    return dHel

def initR():
    
    beta = parameters.beta
    ndof = parameters.ndof

    c,w = initModelParams()


    sigp=np.sqrt( w / ( 2 * np.tanh( 0.5*beta*w ) ) )
    sigq=sigp / w


    R = np.zeros(( ndof ))
    P = np.zeros(( ndof ))

    R0=0
    P0=0.0
    for d in range(ndof):
        #R[d] = R0 + sigq[d] * random.gauss(0,1)
        #P[d] = P0 + sigp[d] * random.gauss(0,1)
        R[d] =  random.gauss(R0,sigq[d])
        P[d] =  random.gauss(R0,sigp[d])
    return R,P

