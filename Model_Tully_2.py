import numpy as np
import time
import numpy.linalg as LA

class parameters():
    NCPUS = 150
    dtI = 0.25
    NSteps = int(1200 / dtI) ## 41.350 a.u. / fs
    NTraj = int(10**5)
    EStep = 50
    dtE = dtI/EStep
    M = 2000
    NStates = 2
    initState = 1
    method = "spin-pldm" # "SQC" or "PLDM" or "spin-pldm"
    sampling = "focused"

    windowtype = "n-triangle" # "Square", "N-Triangle", only for SQC
    adjustedgamma = "yes" # "yes", "no", only for SQC

    dirName = "TRAJ_spin-PLDM_10_pow_5/"

    NSkip = 0.5  # Plot every {} a.u.



    fs_to_au = 41.341 # a.u./fs

def Hel(R):

    N = parameters.NStates

    # Assign to Hamiltonian elements
    Hel = np.zeros((2,2))
    A = 0.10
    B = 0.28
    C = 0.015
    D = 0.06
    E0 = 0.05

    Hel[1,1] = -A * np.exp(-B*R**2) + E0
    Hel[1,0] = C * np.exp( -D*R**2 )
    Hel[0,1] = Hel[1,0]

    return Hel

def dHel0(R):
    return np.zeros((len(R)))

def dHel(R):

    N = parameters.NStates

    # Assign to Hamiltonian elements
    dHel = np.zeros((2,2,1))
    A = 0.10
    B = 0.28
    C = 0.015
    D = 0.06
    #E0 = 0.05

    dHel[1,1,0] = 2 * A * B * R * np.exp(-B*R**2)
    dHel[0,1,0] = -2 * C * D * R * np.exp(-D*R**2)

    return dHel

def initR():
    R0 = -9.0
    P0 = 30
    alpha = 1.0
    sigR = 1.0/np.sqrt(2.0*alpha)
    sigP = np.sqrt(alpha/2.0)

    # Numpy gives bad random numbers
    #R = np.random.normal()*sigR + R0
    #P = np.random.normal()*sigP + P0

    # Random module gives correct randoms for each trajectory
    R = random.gauss(R0, sigR )
    P = random.gauss(P0, sigP )

    return np.array([R]), np.array([P])
