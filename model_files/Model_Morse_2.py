import numpy as np
import time
import numpy.linalg as LA

class parameters():

    model_Name = "Model_Miller_2"

    NCPUS = 100
    dtI = 2.0
    NSteps = int(3500/dtI)
    NTraj = 50 ** 4
    EStep = 50
    dtE = dtI/EStep
    M = 20000
    NStates = 3
    initState = 0

    windowtype = "square" # "square", "triangle", only for SQC
    adjustedgamma = "yes" # "yes", "no", only for SQC

    dirName = "TRAJ_SQC_SY"

    NSkip = 20

def Hel(x):
    #Models from: Coronado, Xing, and Miller, Chem. Phys. Let. 349, 5–6, 2001, 521-529

    N = parameters.NStates

    D =  np.array([0.020,  0.010,  0.003])
    b =  np.array([0.650,  0.400,  0.650])
    Re = np.array([4.500,  4.000,  4.400])
    c =  np.array([0.000,  0.010,  0.020])
    
    A =  np.array([0.005,  0.005])
    Rcross =  np.array([3.660,  3.340])
    a =  np.array([32.00,  32.00])


    # Assign to Hamiltonian elements
    Hel = np.zeros((3,3))
    Hel[0,0] = D[0] * ( 1 - np.exp( -b[0] * (x - Re[0]) ) )**2 + c[0]
    Hel[1,1] = D[1] * ( 1 - np.exp( -b[1] * (x - Re[1]) ) )**2 + c[1]
    Hel[2,2] = D[2] * ( 1 - np.exp( -b[2] * (x - Re[2]) ) )**2 + c[2]

    Hel[1,0] = A[0] * np.exp( -a[0] * (x - Rcross[0])**2 )
    Hel[2,0] = A[1] * np.exp( -a[1] * (x - Rcross[1])**2 )
    Hel[2,1] = 0

    Hel[0,1] = Hel[1,0]
    Hel[0,2] = Hel[2,0]
    Hel[1,2] = Hel[2,1]

    return Hel

def dHel0(R):
    return np.zeros((len(R)))

def dHel(x):

    #Models from: Coronado, Xing, and Miller, Chem. Phys. Let. 349, 5–6, 2001, 521-529

    N = parameters.NStates

    D =  np.array([0.020,  0.010,  0.003])
    b =  np.array([0.650,  0.400,  0.650])
    Re = np.array([4.500,  4.000,  4.400])
    c =  np.array([0.000,  0.010,  0.020])
    
    A =  np.array([0.005,  0.005])
    Rcross =  np.array([3.660,  3.340])
    a =  np.array([32.00,  32.00])


    # Assign to Hamiltonian elements
    dHel = np.zeros((3,3,1))
    dHel[0,0,0] = 2 * D[0] * b[0] * ( 1 - np.exp( -b[0] * (x - Re[0]) ) ) * np.exp( -b[0] * (x - Re[0]) )
    dHel[1,1,0] = 2 * D[1] * b[1] * ( 1 - np.exp( -b[1] * (x - Re[1]) ) ) * np.exp( -b[1] * (x - Re[1]) )
    dHel[2,2,0] = 2 * D[2] * b[2] * ( 1 - np.exp( -b[2] * (x - Re[2]) ) ) * np.exp( -b[2] * (x - Re[2]) )

    dHel[1,0,0] = -2 * a[0] * A[0] * np.exp( -a[0] * (x - Rcross[0])**2 ) * ( x - Rcross[0] )
    dHel[2,0,0] = -2 * a[1] * A[1] * np.exp( -a[1] * (x - Rcross[1])**2 ) * ( x - Rcross[1] )
    dHel[2,1,0] = 0

    dHel[0,1,0] = dHel[1,0,0]
    dHel[0,2,0] = dHel[2,0,0]
    dHel[1,2,0] = dHel[2,1,0]

    return dHel

def initR():
    R0 = 3.300
    P0 = 0

    omega_c = 5*10**(-3)
    rmass = 1/parameters.M
    
    sigP = np.sqrt( omega_c/(2.0*rmass) )
    sigR = np.sqrt( rmass/(2.0*omega_c) )

    # Numpy gives bad random numbers
    #R = np.random.normal()*sigR + R0
    #P = np.random.normal()*sigP + P0

    # Random module gives correct randoms for each trajectory
    R = random.gauss(R0, sigR )
    P = random.gauss(P0, sigP )

    return R, P


