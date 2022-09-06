import numpy as np
import time
import numpy.linalg as LA

class parameters():
    NCPUS = 100
    dtI = 1.0
    NSteps = int(500 * 41.341/dtI)
    NTraj = 10 ** 4
    EStep = 50
    dtE = dtI/EStep

    # SQC OPTIONS
    windowtype = "square" # "square", "triangle", only for SQC
    adjustedgamma = "yes" # "yes", "no", only for SQC

    dirName = "TRAJ_SQC_SY"

    NSkip = 20


    # MODEL-SPECIFIC ITEMS

    NStates = 2
    initState = 1
    ndof = 3

    conv = 27.2114 # eV / a.u.

    # VAR      STATE 1       STATE 2
    Ek    = [ 3.940/conv,  4.840/conv  ]
    w1    = [ 0.126/conv,  0.126/conv  ]
    kap1  = [ 0.037/conv,  -0.254/conv ]
    w6a   = [ 0.074/conv,  0.074/conv  ]
    kap6a = [ -0.105/conv, 0.149/conv  ]
    w10a  = [ 0.118/conv,  0.118/conv  ]
    lam = 0.262/conv

    w_m = np.array([ w1[0], w6a[0], w10a[0] ])
    M = np.array([  1/w1[0], 1/w6a[0], 1/w10a[0]  ]) # 3 Modes = 3 Masses

def Hel(R):    # state-dependent potential

    Ek      = parameters.Ek
    kap1    = parameters.kap1
    kap6a   = parameters.kap6a
    lam     = parameters.lam
    NStates = parameters.NStates
    
    VMat = np.zeros(( NStates, NStates ))

    # diagonal coupling

    VMat[0,0] = Ek[0] + kap1[0] * R[0] + kap6a[0] * R[1]    # should not have 0.5 * ( w1[0] * R[0]**2 + w6a[0] * R[1]**2 + w10a[0] * R[2]**2 )  
    VMat[1,1] = Ek[1] + kap1[1] * R[0] + kap6a[1] * R[1]    # since quadratic in position has been included in the Nuclei sampling

    # off-diagonal coupling

    VMat[0,1] = lam * R[2]          # Lambda has no state-dependence or mode dependence. Is this correct? -- yes, and only R2 is associated with lambda
    VMat[1,0] = VMat[0,1]     # Symmetric Hamiltonian

    return VMat



def dHel0(R):   # derivative of state-independent potential

    ndof = parameters.ndof
    w    = parameters.w_m

    dVMat0 = np.zeros(( ndof ))
    
    for k in range(ndof):
        dVMat0[k] = w[k] * R[k]

    return dVMat0



def dHel(R):    # derivative of state-dependent potential

    kap1    = parameters.kap1
    kap6a   = parameters.kap6a
    lam     = parameters.lam
    NStates = parameters.NStates
    ndof    = parameters.ndof

    dVMat = np.zeros(( NStates, NStates, ndof ))

    dVMat[0,0,0] = kap1[0]  # First Mode
    dVMat[0,0,1] = kap6a[0] # Second Mode

    dVMat[0,1,2] = lam      # only x_10a the coupling coordinate is attached to this lambda, but it is not mentioned in the paper...
    dVMat[1,0,2] = lam      # Duncan and I had checked the model in previous papers that they had cited.

    dVMat[1,1,0] = kap1[1]  # First Mode
    dVMat[1,1,1] = kap6a[1] # Second Mode

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
