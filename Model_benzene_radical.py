import numpy as np

class parameters():

    NCPUS = 50
    dtI = 0.5 # Nuclear Time Step, a.u.
    T = 300 * 41.341 # Length of Simulation, a.u.
    NSteps = int( T / dtI ) ## 41.350 a.u. / fs
    NTraj = 10 ** 5
    EStep = 50
    dtE = dtI/EStep

    # SQC OPTIONS
    windowtype = "langer-correction" # "square", "triangle", "langer-correction"
    adjustedgamma = "no" # "yes", "no"

    dirName = "TRAJ_SQC_LN"

    NSkip = 10  # Plot every {NSkip} nuclear steps




    # MODEL-SPECIFIC ITEMS

    NStates = 3
    initState = 2
    ndof = 5

    AUtoEV = 27.2114 # eV / a.u.

    # VAR                 STATE 1     STATE 2        STATE 3
    Ek      = np.array([  9.750/AUtoEV,  11.84/AUtoEV,  12.44/AUtoEV  ])
    k2      = np.array([ -0.042/AUtoEV, -0.042/AUtoEV, -0.301/AUtoEV  ])
    k16     = np.array([ -0.246/AUtoEV,  0.242/AUtoEV,  0.000/AUtoEV  ])
    k18     = np.array([ -0.125/AUtoEV,  0.100/AUtoEV,  0.000/AUtoEV  ])
    k8      = np.zeros(NStates)
    k19     = np.zeros(NStates)

    k = np.c_[k2,k16,k18,k8,k19]

    w2      = np.ones(NStates) * 0.123/AUtoEV
    w16     = np.ones(NStates) * 0.198/AUtoEV
    w18     = np.ones(NStates) * 0.075/AUtoEV
    w8      = np.ones(NStates) * 0.088/AUtoEV
    w19     = np.ones(NStates) * 0.120/AUtoEV

    lam8 = 0.164/AUtoEV
    lam19 = 0.154/AUtoEV

    w_m = np.array([ w2[0], w16[0], w18[0], w8[0], w19[0] ])
    M = np.array([  1/w2[0], 1/w16[0], 1/w18[0], 1/w8[0], 1/w19[0]  ]) # 5 Modes = 5 Masses

def Hel(R):    # state-dependent potential

    Ek      = parameters.Ek     # 3
    k       = parameters.k      # 3x5
    w_m     = parameters.w_m    # 5
    lam8    = parameters.lam8   # 1
    lam19   = parameters.lam19  # 1

    NStates = parameters.NStates
    VMat = np.zeros(( NStates, NStates ))

    # diagonal coupling

    VMat[0,0] = Ek[0] + np.sum( k[0,:] * R[:] )
    VMat[1,1] = Ek[1] + np.sum( k[1,:] * R[:] )
    VMat[2,2] = Ek[2] + np.sum( k[2,:] * R[:] )

    # off-diagonal coupling

    VMat[0,1] = lam8  * R[3] # Attached to fourth mode
    VMat[1,2] = lam19 * R[4] # Attached to fifth mode
    
    VMat[1,0] = VMat[0,1]  # Symmetric Hamiltonian
    VMat[2,1] = VMat[1,2]  # Symmetric Hamiltonian

    return VMat



def dHel0(R):   # derivative of state-independent potential

    w_m     = parameters.w_m
    ndof    = parameters.ndof
    dVMat0  = np.zeros(( ndof ))
    
    for k in range(ndof):
        dVMat0[k] = w_m[k] * R[k]
    return dVMat0



def dHel(R):    # derivative of state-dependent potential

    Ek      = parameters.Ek
    w_m     = parameters.w_m
    lam8    = parameters.lam8
    lam19   = parameters.lam19

    k = parameters.k

    NStates = parameters.NStates
    ndof = parameters.ndof

    dVMat = np.zeros(( NStates, NStates, ndof ))

    dVMat[0,0,:] = k[0,:]
    dVMat[1,1,:] = k[1,:]
    dVMat[2,2,:] = k[2,:]

    dVMat[0,1,3] = lam8
    dVMat[1,2,4] = lam19

    return dVMat



def initR():     # Initial product state of the vibrational ground state <\Psi| = \Product_j \pi^{-1/4} \exp{ -x_j^2 / 2 }

    ndof    = parameters.ndof
    
    R = np.zeros( ndof )
    P = np.zeros( ndof )  

    sigP = np.ones( ndof )
    sigR = np.ones( ndof )

    for d in range( ndof ):
        R[d] = np.random.normal() * sigR[d]       # np.random.normal() gives exp( -x**2 / 2) distribution
        P[d] = np.random.normal() * sigP[d]  
        
    return R, P
