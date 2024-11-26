import numpy as np
from numba import jit
import model as m
from model import H, dH, dH0, initR # import the functions from the model
import random

def get_Globals():
    global NR, NStates, dt, NSkip, NTraj, NSteps, NStepsPrint, M, initState, initBasis
    NR          = m.NR
    NStates     = m.NStates
    dt          = m.dt
    NTraj       = m.NTraj
    NSteps      = m.NSteps
    M           = m.M
    initState   = m.initState
    initBasis   = m.initBasis

    global alpha, beta#, ZPE
    sumN  = np.sum(np.array([1/n for n in range(1,NStates+1)]))
    alpha = (NStates - 1)/(sumN - 1)
    beta  = (alpha - 1)/NStates
    #ZPE   = beta/alpha # This is not used in the code

@jit(nopython=True)
def initc( U ): # initialization of the coefficients
    z = np.sqrt(beta/alpha) * np.ones((NStates), dtype = np.complex128)
    z[initState] = np.sqrt( (1+beta)/alpha )
    # for n in range(NStates):
    #     rand = random.random()
    #     z[n] = z[n] * np.exp(1j * 2 * np.pi * rand)
    z = z * np.exp( 1j * 2 * np.pi * np.random.uniform(0,1,size=NStates) )
    if(initBasis==2): # initState provided was for diabatic basis
        z[:] = np.conj(U).T @ z # Rotate to adiabatic basis
    return z

   
@jit(nopython=True)
def createEU( R ):
    """
    Choose overlap matrix to be consistent with identity matrix at t = 0
    """
    E, U = np.linalg.eigh(H(R))
    overlap_mat = np.conj( U ).T
    for n in range(NStates):
        f = np.exp(1j*np.angle(overlap_mat[n,n]))
        U[:,n] = f * U[:,n] # phase correction
    return E, U
    
@jit(nopython=True)
def updateEU( R, U0 ):
    """
    Phase consistency scheme similar to A. Akimov
    """
    E1, U1 = np.linalg.eigh( H(R) )
    overlap_mat = np.conj(U1).T @ U0
    for n in range(NStates):
        f = np.exp(1j * np.angle(overlap_mat[n,n]) )
        U1[:,n] = f * U1[:,n] # phase correction
    return E1, U1
        
@jit(nopython=True)
def DtoA_2D(matD, U): # convert operator from diabatic to adiabatic basis
    return np.conj(U).T @ matD @ U

@jit(nopython=True)
def DtoA_3D(matD, U): # convert operator from diabatic to adiabatic basis
    matA = np.zeros(np.shape(matD), dtype = np.complex128)
    for i in range(len(matD[:,0,0])):
        matA[i] = np.conj(U).T @ matD[i] @ U
    return matA
        
@jit(nopython=True)
def updateForce( R, U, AS ): # this calculates the classical force on each nuclear DOF using only the active state
    F     = -dH0(R) # state-independent force
    F[:] += -DtoA_3D(dH(R), U)[:,AS,AS] # state-dependent force for diabatic dH
    return F.real

@jit(nopython=True)
def Velocity_Verlet(R, V, z, AS, U0, F0):
    # Nuclear propagation
    R[:]     += V * dt + 0.5 * F0 / M * dt**2
    E1, U1    = updateEU( R, U0 )
    F1        = updateForce( R, U0, AS )
    V[:]     += 0.5 * dt * (F0 + F1) / M
    # Electronic propagation
    z[:]      = np.conj(U1).T @ U0 @ z # |E(t1)><D(t1)| |E(t0)><D(t0)| |D(t0)>
    z[:]      = np.exp(-1j * dt * E1) * z # Apply phase factor to wavefunction in diagonal basis
    return R, V, z, E1, U1, F1
        
@jit(nopython=True)
def getHopDir(z, R, E, U, AS, AS_proposed): # calculate the direction of the hop from AS to AS_proposed
    """
    BMW: This function can be optimized. 
    BMW: Only need last three lines for ab initio, since NACV already exists.
    """
    hop_dir = np.zeros( NR )
    dc      = np.zeros((NR, NStates, NStates), dtype=np.complex128)
    dE      = np.expand_dims(E, axis=1) - E # BMW: Is this the same as E[:,None] - E[None,:] ???
    for i in range(NStates):
        dE[i,i] = 1.0
    for i in range(NR):
        dc[i,:,:] = DtoA_2D(dH(R)[i], U) / dE # BMW: Why do we call dH many times...?
        for j in range(NStates):
            dc[i,j,j] = 0.0 # Set diagonal NACs to zero.
    ##### BELOW ARE THE IMPORTANT LINES #####
    for k in range(NR):
        for n in range(NStates): # general rescaling direction, slow
            hop_dir[k] += np.real(dc[k,n,AS]         *z[AS]         *np.conj(z[n]) \
                                 -dc[k,n,AS_proposed]*z[AS_proposed]*np.conj(z[n])) / np.sqrt(M[k]) 
    ########################################
    return hop_dir
        
@jit(nopython=True)
def hop( z, R, P, E, U, AS, AS_proposed ): # attempt a hop
    P[:]            = P/np.sqrt(M) # mass-weighted momentum
    potdiff         = np.real( E[AS_proposed] - E[AS] )
    hop_dir         = getHopDir( z, R, E, U, AS, AS_proposed )
    P_proj          = np.dot(P,hop_dir) * hop_dir / np.dot(hop_dir,hop_dir) # projected P along hop_dir
    P_proj_norm     = np.linalg.norm(P_proj) # np.sqrt(np.dot(P_proj,P_proj))
    P_orth          = P - P_proj # orthogonal P
    if( P_proj_norm**2 < 2*potdiff ): # Reject Hop
        accepted = False
        P_proj      = -P_proj # reverse projected momentum
        P[:]        = P_orth + P_proj
        P[:]        = P*np.sqrt(M)
    else: # Accept Hop
        accepted = True
        P_proj      = np.sqrt(P_proj_norm**2 - 2*potdiff)/P_proj_norm * P_proj # scale projected momentum
        P[:]        = P_orth + P_proj
        P[:]        = P*np.sqrt(M)
    return accepted, P



@jit(nopython=True)
def pop_diff( z, AS ): # return population difference between current active state and next highest populated state
    pop = np.abs(z)**2
    pop[AS] = 0.0
    diff = np.abs(z[AS])**2 - np.max(pop)
    if( np.abs(diff) > 10**(-15) ):
        return diff
    else:
        return 0.0 # return 0 if difference is too small (avoids floating point issues)
    
@jit(nopython=True)
def getACST_proposal(z, AS):
    pop         = np.abs(z)**2
    pop[AS]     = 0.0
    AS_proposed = np.argmax(pop)
    return AS_proposed
    
@jit(nopython=True)
def rho( z ): # returns the density matrix estimator (populations and coherences)
    return alpha * np.outer(z,np.conj(z)) - beta * np.identity(NStates) # works in any basis

@jit(nopython=True)
def getACST( z ): # return state with largest population
    AS = np.argmax( np.abs(z) )
    return AS
        
@jit(nopython=True)   
def check_for_hop( z, R, P, E, U, AS, F ): # determine rescaling conditions and perform hop attempt
    if( pop_diff( z, AS ) <= 0.0 ): 
        AS_proposed = getACST_proposal( z, AS )
        accepted, P = hop( z, R, P, E, U, AS, AS_proposed )
        if ( accepted ):
            AS = 1 * AS_proposed
            F = updateForce( R, U, AS )
    return AS, F

def runTraj():
    
    # Create output arrays
    active_state = np.zeros((NTraj,          NSteps+1)) # stores active state for each (printed) timestep and for each trajectory
    rho_A        = np.zeros((NTraj,  NStates,NStates,NSteps+1), dtype=np.complex128)
    rho_D        = np.zeros((NTraj,  NStates,NStates,NSteps+1), dtype=np.complex128)
    
    for itraj in range(NTraj): # repeat simulation for each trajectory
        print("Trajectory %d of %d " % (itraj, NTraj))
        
        # Initialize trajectory information
        R,P   = initR()              # initialize nuclear positions and momenta
        V     = P/M                  # convert momentum to velocity
        E0,U0 = createEU( R )        # initialize electronic energies and eigenvectors
        z     = initc( U0 )          # initialize state wavefunction
        AS    = getACST( z )         # initialize active state
        F0    = updateForce( R, U0, AS ) # initialize classical force
        
        for step in range(NSteps): # single trajectory
            #print("Time: ", t*dt)
            
            # Save output variables
            active_state[itraj, step]  = AS # store active state
            rho_A[itraj,:,:,step]      = rho(z)
            rho_D[itraj,:,:,step]      = rho(U0 @ z)
                
            # Do full timestep
            R, V, z, E0, U0, F0 = Velocity_Verlet( R, V, z, AS, U0, F0 )
            AS, F0              = check_for_hop( z, R, M*V, E0, U0, AS, F0 )
            
        # Save output variables for last time
        active_state[itraj,-1]  = AS # store active state
        rho_A[itraj,:,:,-1]     = rho(z)
        rho_D[itraj,:,:,-1]     = rho(U0 @ z)
        
    # Return output variables
    time = np.linspace(0, NSteps*dt, NSteps+1)
    return time, active_state, rho_A, rho_D

if (__name__ == "__main__"):
    get_Globals()
    time, active_state, rho_A, rho_D = runTraj()

    # Average over trajectories
    active_state = np.average(active_state, axis=0)
    rho_A        = np.average(rho_A, axis=0)
    rho_D        = np.average(rho_D, axis=0)

    # Get exact results
    from model import exact_results
    exact_results = exact_results() # time, P0 (Dia.), P1 (Dia.)

    from matplotlib import pyplot as plt
    for state in range(NStates):
        #plt.plot( exact_results[:,0], exact_results[:,state+1], c='black', lw=6, alpha=0.5, label="Exact P$_%d$ (Dia.)"%(state) )
        plt.plot( exact_results[:,0], exact_results[:,state+1+2*NStates], c='black', lw=6, alpha=0.5, label="Exact P$_%d$ (Ad.)"%(state) )

    #time = np.linspace(0, exact_results[-1,0], len(active_state))
    plt.plot(time, active_state, label="AS")
    for state in range(NStates):
        #plt.plot(time, rho_D[state,state].real, c='red', lw=2, label="P$_%d$ (Dia.)"%(state))
        plt.plot(time, rho_A[state,state].real, c='red', lw=1, label="P$_%d$ (Ad.)"%(state))
    
    plt.legend()
    plt.xlim(0, time[-1])
    plt.ylim(0,1)
    plt.xlabel("Time (a.u.)", fontsize=15)
    plt.ylabel("Population", fontsize=15)
    #plt.xlim(time[0], time[-1])
    plt.savefig("test.jpg", dpi=300)
