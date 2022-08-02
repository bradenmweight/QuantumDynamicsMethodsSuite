import numpy as np
from scipy.integrate import solve_ivp
import multiprocessing as mp
import time, os, sys
import Model_Miller_3 as model
import numpy.linalg as LA
import subprocess as sp
from random import random

def getGlobalParams():
    global dtE, dtI, NSteps, NTraj, NStates, M
    global NCPUS, initState, dirName
    global fs_to_au, NSkip, NDOF
    global rescale_type, decoherece_type
    dtE = model.parameters.dtE
    dtI = model.parameters.dtI
    NSteps = model.parameters.NSteps
    NTraj = model.parameters.NTraj
    NStates = model.parameters.NStates
    M = model.parameters.M
    NCPUS = model.parameters.NCPUS
    initState = model.parameters.initState # Initial Adiabatic State
    dirName = model.parameters.dirName
    rescale_type = model.parameters.rescale_type
    NSkip = model.parameters.NSkip
    NDOF = model.parameters.NDOF
    decoherece_type = model.parameters.decoherece_type
    fs_to_au = 41.341 # a.u./fs

def cleanMainDir():
    if ( os.path.exists(dirName) ):
        sp.call("rm -r "+dirName,shell=True)
    sp.call("mkdir "+dirName,shell=True)

def cleanDir(n):
    if ( os.path.exists( dirName + "/traj-"+str(n) ) ):
        sp.call("rm -r "+dirName+"/traj-"+str(n),shell=True)
    sp.call("mkdir "+dirName+"/traj-"+str(n),shell=True)

def initFiles(n):
    activeStateFile = open(dirName+"/traj-" + str(n) + "/active_state.dat","w")
    energyFileAd = open(dirName+"/traj-" + str(n) + "/energy_ad.dat","w") # Kinetic, Potential, Total
    energyFileDia = open(dirName+"/traj-" + str(n) + "/energy_dia.dat","w") # Kinetic, Potential, Total
    densityFiles = [ open(dirName+"/traj-" + str(n) + "/density_ad_re.dat","w"), \
                     open(dirName+"/traj-" + str(n) + "/density_ad_im.dat","w"), \
                     open(dirName+"/traj-" + str(n) + "/density_dia_re.dat","w"), \
                     open(dirName+"/traj-" + str(n) + "/density_dia_im.dat","w") ]
    InitCondsFile = open(dirName+"/traj-" + str(n) + "/initconds.dat","w")
    RFile = open(dirName+"/traj-" + str(n) + "/RFile.dat","w")
    HelFile = open(dirName+"/traj-" + str(n) + "/Hel.dat","w")
    return activeStateFile, energyFileAd, energyFileDia, densityFiles, InitCondsFile, RFile, HelFile

def closeFiles(activeStateFile, energyFileAd, energyFileDia, densityFiles, InitCondsFile, RFile, HelFile):
    energyFileAd.close()    
    energyFileDia.close()    
    densityFiles[0].close()    
    densityFiles[1].close()    
    densityFiles[2].close()   
    densityFiles[3].close()   
    InitCondsFile.close()
    RFile.close()
    HelFile.close()

def writeDensity(densityFiles,z_ad,z_dia,active_state,activeStateFile,step):

    rho_ad = get_density_matrix( z_ad )
    rho_dia = get_density_matrix( z_dia )

    outArrayRE = [ step * dtI, step * dtI / fs_to_au ]
    outArrayIM = [ step * dtI, step * dtI / fs_to_au ]

    # Write adiabatic density matrix
    elements = rho_ad.reshape(-1)
    for element in elements:
        outArrayRE.append( np.real( element ) )
        outArrayIM.append( np.imag( element ) )

    densityFiles[0].write( "\t".join(map("{:.5f}".format,outArrayRE)) + "\n")
    densityFiles[1].write( "\t".join(map("{:.5f}".format,outArrayIM)) + "\n")

    # Write diabatic density matrix
    elements = rho_dia.reshape(-1)
    for element in elements:
        outArrayRE.append( np.real( element ) )
        outArrayIM.append( np.imag( element ) )

    densityFiles[2].write( "\t".join(map("{:.5f}".format,outArrayRE)) + "\n")
    densityFiles[3].write( "\t".join(map("{:.5f}".format,outArrayIM)) + "\n")

    # Write adiabatic active state population
    active_pop = np.zeros(( NStates ))
    active_pop[active_state] = 1
    activeStateFile.write( "{:.5f}\t{:.5f}".format(step * dtI, step * dtI / fs_to_au) + "\t" + "\t".join(map("{:.5f}".format,active_pop)) + "\n" )

    return None

def writeEnergy_ad(energyFileAd,active_state,step,M,V,Ead):

    KIN = get_Kinetic( M, V )
    POT = get_Potential( Ead, active_state )
    TOT = KIN + POT

    energyFileAd.write( "{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\n".format(step * dtI, step * dtI / fs_to_au, KIN, POT, TOT) )

def writeEnergy_dia(energyFileDia,active_state,step,M,V,Hel):

    KIN = get_Kinetic( M, V )
    POT = Hel[active_state, active_state]
    TOT = KIN + POT

    energyFileDia.write( "{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\n".format(step * dtI, step * dtI / fs_to_au, KIN, POT, TOT) )

def writeR(R,RFile):
    RFile.write(str(round(R[0],4)) + "\n")

def writeHel(Hel,HelFile):
    outList = []
    for i in range(NStates):
        for j in range(NStates):
            outList.append( round(Hel[i,j],6) )
    HelFile.write( "\t".join(map(str,outList))+"\n" )

def initCoeffs(InitCondsFile):# Initialization of the mapping Variables
    """
    Returns np.array((F)) of adiabatic electronic coefficients z_j
    """
    z = np.zeros( ( NStates ), dtype=complex )
    z[initState] = 1.0
    active_state = initState
    return z, active_state

def get_density_matrix( z ):
    """
    Returns np.array((F,F)) of electronic density matrix rho_jk
    """
    rho = np.outer( z , np.conjugate(z) )
    return  rho

def prop_el_coeffs_VV(z_ad, Ead, deriv_coup, V, dtE, ESteps):
    """
    Propagates electornic coefficients
    """

    Zreal = np.real(z_ad)
    Zimag = np.imag(z_ad)

    NACT = get_scalar_coupling( V, deriv_coup )
    E_mat = np.diag(Ead)

    for step in range( ESteps ):

        # Propagate Imaginary first by dt/2
        Zimag[:] += -1.000 * Ead[:] * Zreal[:] * dtE/2 - NACT[:,:] @ Zimag[:] * dtE/2

        # Propagate Real by full dt
        Zreal[:] +=  1.000 * Ead[:] * Zimag[:] * dtE   - NACT[:,:] @ Zreal[:] * dtE
        
        # Propagate Imaginary final by dt/2
        Zimag[:] += -1.000 * Ead[:] * Zreal[:] * dtE/2 - NACT[:,:] @ Zimag[:] * dtE/2


    return  Zreal + 1j*Zimag

def prop_el_coeffs_RK_SCIPY(z_ad, Ead, deriv_coup, V, dtE, ESteps):
    
    def ODE( t, z_ad, E_mat, NACT ):
        return -1j * E_mat @ z_ad - NACT @ z_ad

    NACT = np.einsum("d,JKd->JK", V, deriv_coup) # Scalar Coupling
    E_mat = np.diag(Ead)

    z_ad_0 = np.copy(z_ad)
    sol = solve_ivp(ODE, t_span=[0,ESteps*dtE/2], y0=z_ad_0, t_eval=[ESteps*dtE/2], max_step=dtE, method="DOP853", args=(E_mat, NACT))

    z_ad = sol.y[:,-1]

    return z_ad

def Force(dHel_ad, R, z, active_state, dHel0):
    """
    F = F0 (State-Ind.) + Fad (Hell.-Feyn.)
    Force only contributes from the active ADIABATIC state
    """

    F = np.zeros(( len(R) ))
    F -= dHel_ad[active_state,active_state,:] + dHel0
    return F

def get_derv_coupling( dHel0, dHel, Uad, Ead ):
    """
    d = <j| \grad H |k>
        _______________
           E_k  - E_j
    |j>, |k> are adiabatic states
    \grad H = \grad ( H_0 + H_ss )
    """

    Ediffs = np.subtract.outer(Ead,Ead)
    Ediffs[np.diag_indices(len(Ediffs))] = 1.0 # Replace zeros on diagonal with "1"

    deriv_coup = np.zeros(( NStates, NStates, NDOF ))
    for d in range( NDOF ):
        deriv_coup[:,:,d] = Uad.T @ (dHel0 + dHel[:,:,d]) @ Uad / Ediffs
    
    deriv_coup[np.diag_indices(len(deriv_coup))] = 0.0 # Replace diagonal with "0"

    return deriv_coup 

def get_scalar_coupling( V, deriv_coup ):
    return np.einsum( "d,ijd->ij", V, np.real(deriv_coup) )

def get_hop_prob( z_ad, V, deriv_coup, active_state, dtI ):
    
    rho = get_density_matrix( z_ad )
    POP = np.real( z_ad * np.conjugate(z_ad) )
    NACT = get_scalar_coupling( V, deriv_coup )

    probs = np.zeros(( NStates ))
    for j in range( NStates ):
        if ( j != active_state ):
            b = -2 * np.real(rho[j,active_state]) * NACT[j,active_state]
            probs[j] = b * dtI / POP[active_state]

            if ( probs[j] < 0.0 ):
                probs[j] = 0.0

    return probs

def hop_check( active_state, probs ):

    rand = random()
    hop_data = [False, active_state, active_state]
    for j in range( NStates ):
        if ( j == active_state ):
            continue
        if ( np.sum(probs[:j]) < rand and rand <= np.sum(probs[:j+1]) ):
            hop_data[0] = True
            hop_data[1] = active_state
            hop_data[2] = j

    #print(hop_data)
    return hop_data

def get_Kinetic( M, V ):
    return 0.5000 * M * np.einsum( "d,d->", V, V)

def get_Potential( Ead, active_state ):
    return Ead[active_state]

def get_Potential_Diff( Ead, hop_data ):
    start = hop_data[1]
    end = hop_data[2]
    return Ead[end] - Ead[start]

def evaluate_hop( active_state, hop_data, M, V, Ead, deriv_coup, z_ad ):

    if ( hop_data[0] == True ):

        KIN = get_Kinetic( M, V )
        POT = get_Potential( Ead, active_state )
        PDIFF = get_Potential_Diff( Ead, hop_data )

        start_state = hop_data[1]
        end_state = hop_data[2]

        # Accept or Reject Hop Based on Kinetic Energy
        
        if ( rescale_type == 'energy' ):
            if ( KIN < PDIFF ):
                # Rejecting hop
                print("REJECTING HOP BASED ON LACK OF K.E.:")
                return V, active_state

            # Calculate the uniform scaling factor based on energy criterion
            scale_factor = np.sqrt( 1 - PDIFF / KIN )
            V *= scale_factor

            # Rearrange states based on hop
            active_state = end_state
            tmp = np.copy( z_ad[start_state] )
            z_ad[start_state] = z_ad[end_state]
            z_ad[end_state] = tmp

        else:
            assert(False), "'rescale_type' is only coded for 'energy'"


        #### DECOHERENCE CORRECTIONS ####
        if ( decoherece_type == "IDC" ):
            #print("Implementing IDC decoherence correction.")
            z_ad = np.zeros((NStates), dtype=complex)
            z_ad[active_state] = 1.0 + 0.0j



    return V, active_state



def VelVerF(R, P, z_ad, active_state, RFile, HelFile, energyFileAd, energyFileDia, densityFiles, activeStateFile, step): # Ionic position, ionic momentum, etc.
    
    Hel = model.Hel(R) # Electronic Structure
    Ead, Uad = np.linalg.eigh(Hel) # Return adiabatic states and transformation matrices
    dHel = model.dHel(R) # Diabatic forces
    dHel_ad = np.einsum("IJ,JKd,KL->ILd", Uad.T, dHel, Uad) # Transform forces to adiabatic representation
    dHel0 = model.dHel0(R) # This is scalar. No need to transform.
    deriv_coup = get_derv_coupling( dHel0, dHel, Uad, Ead )
    z_dia = Uad @ z_ad

    if ( step % NSkip == 0 ): 
        writeDensity(densityFiles,z_ad,z_dia,active_state,activeStateFile,step)
        writeEnergy_ad(energyFileAd,active_state,step,M,P/M,Ead)
        #writeEnergy_dia(energyFileDia,active_state,step,M,P/M,Hel)

    ESteps = int(dtI/dtE)
    z_ad = prop_el_coeffs_VV(z_ad, Ead, deriv_coup, P/M, dtE, ESteps//2)
    #z_ad = prop_el_coeffs_RK_SCIPY(z_ad, Ead, deriv_coup, P/M, dtE, ESteps//2)

    # FSSH Hopping
    probs = get_hop_prob(z_ad, P/M, deriv_coup, active_state, dtI)
    hop_data = hop_check(active_state, probs)
    V, active_state = evaluate_hop(active_state, hop_data, M, P/M, Ead, deriv_coup, z_ad) # Put Decoherence correction here # TODO

    F1 = Force(dHel_ad, R, z_ad, active_state, dHel0)

    P += 0.5000 * F1 * dtI # Half-step velocity

    R += P / M * dtI # Full Step Position

    Hel = model.Hel(R) # Electronic Structure
    Ead, Uad = np.linalg.eigh(Hel) # Return adiabatic states and transformation matrices
    dHel = model.dHel(R) # Diabatic forces
    dHel_ad = np.einsum("IJ,JKd,KL->ILd", Uad.T, dHel, Uad) # Transform forces to adiabatic representation
    dHel0 = model.dHel0(R) # This is scalar. No need to transform.
    deriv_coup = get_derv_coupling( dHel0, dHel, Uad, Ead )
    z_dia = Uad @ z_ad

    F2 = Force(dHel_ad, R, z_ad, active_state, dHel0)
    
    P += 0.5000 * F2 * dtI # Half-step Velocity

    z_ad = prop_el_coeffs_VV(z_ad, Ead, deriv_coup, P/M, dtE, ESteps//2)
    #z_ad = prop_el_coeffs_RK_SCIPY(z_ad, Ead, deriv_coup, P/M, dtE, ESteps//2)
    
    return R, P, z_ad, active_state

def RunIterations(n): # This is parallelized already. "Main" for each trajectory.

    cleanDir(n)
    activeStateFile, energyFileAd, energyFileDia, densityFiles, InitCondsFile, RFile, HelFile = initFiles(n) # Makes file objects

    R,P = model.initR() # Initialize nuclear DOF

    z, active_state = initCoeffs(InitCondsFile)

    for step in range(NSteps):
        #print ("Step:", step, active_state)
        R, P, z, active_state = VelVerF(R, P, z, active_state, RFile, HelFile, energyFileAd, energyFileDia, densityFiles, activeStateFile, step)
    
    closeFiles(activeStateFile, energyFileAd, energyFileDia, densityFiles, InitCondsFile, RFile, HelFile)

### Start Main Program ###
if ( __name__ == "__main__"  ):

    getGlobalParams()
    cleanMainDir()
    start = time.time()

    print (f"There will be {NCPUS} cores with {NTraj} trajectories.")

    runList = np.arange(NTraj)
    with mp.Pool(processes=NCPUS) as pool:
        pool.map(RunIterations,runList)

    stop = time.time()
    print (f"Total Computation Time (Hours): {(stop - start) / 3600}")


