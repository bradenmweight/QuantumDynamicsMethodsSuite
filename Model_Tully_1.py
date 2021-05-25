import numpy as np
import time
import numpy.linalg as LA

class parameters():
    NCPUS = 200
    dtI = 0.1
    NSteps = int(1200 / dtI) ## 41.350 a.u. / fs
    NTraj = 10000
    EStep = 50
    dtE = dtI/EStep
    M = 2000
    NStates = 2
    initState = 1
    method = "spin-pldm" # "SQC" or "PLDM" or "spin-pldm"
    sampling = "focused"

    windowtype = "n-triangle" # "Square", "N-Triangle", only for SQC
    adjustedgamma = "yes" # "yes", "no", only for SQC

    dirName = "TRAJ_spin-PLDM_10000"

    NSkip = 0.5  # Plot every {} a.u.



    fs_to_au = 41.341 # a.u./fs

def Hel(R):

    N = parameters.NStates

    # Assign to Hamiltonian elements
    Hel = np.zeros((2,2))
    A = 0.01
    B = 1.6
    C = 0.005
    D = 1.0

    if ( R > 0 ):
        Hel[0,0] = A * ( 1 - np.exp(-B*R) )
        Hel[1,1] = -Hel[0,0]
    else:
        Hel[0,0] = -A * ( 1 - np.exp(B*R) )
        Hel[1,1] = -Hel[0,0]

    Hel[1,0] = C * np.exp( -D*R**2 )
    Hel[0,1] = Hel[1,0]

    

    return Hel

def dHel0(R):
    return np.zeros((len(R)))

def dHel(R):

    N = parameters.NStates

    # Assign to Hamiltonian elements
    dHel = np.zeros((2,2,1))
    A = 0.01
    B = 1.6
    C = 0.005
    D = 1.0

    if ( R > 0 ):
        dHel[0,0,0] = A * B * np.exp( -B*R )
    else:
        dHel[0,0,0] = A * B * np.exp( B*R )

    dHel[0,1,0] = -2 * C * D * R * np.exp(-D*R**2)
    dHel[1,0,0] = dHel[0,1,0]

    return dHel

def initR():
    R0 = -9.0
    P0 = 30
    alpha = 1.0
    sigR = 1.0/np.sqrt(2.0*alpha)
    sigP = np.sqrt(alpha/2.0)

    R = np.random.normal()*sigR + R0
    P = np.random.normal()*sigP + P0

    return np.array([R]), np.array([P])
