import numpy as np
from scipy.integrate import solve_ivp
import multiprocessing as mp
import time, os, sys
import Model_Miller_3 as model
#import Model_Tully_1 as model
import numpy.linalg as LA
import subprocess as sp
from random import random
from numba import jit

def getGlobalParams():
    global dtE, dtI, NSteps, NTraj, NStates, M
    global NCPUS, initState, dirName
    global fs_to_au, NSkip, NDOF
    global rescale_type, decoherece_type, SWAP_COEFFS_HOP
    global Diabatic_Density_Type, EDC_PARAM, AS_POP_INCREASE
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
    SWAP_COEFFS_HOP = model.parameters.SWAP_COEFFS_HOP
    decoherece_type = model.parameters.decoherece_type
    Diabatic_Density_Type = model.parameters.Diabatic_Density_Type
    EDC_PARAM = model.parameters.EDC_PARAM
    AS_POP_INCREASE = model.parameters.AS_POP_INCREASE
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
    VFile = open(dirName+"/traj-" + str(n) + "/VFile.dat","w")
    HelFile = open(dirName+"/traj-" + str(n) + "/Hel.dat","w")
    probFile = open(dirName+"/traj-" + str(n) + "/Prob.dat","w")
    randFile = open(dirName+"/traj-" + str(n) + "/Rand.dat","w")
    NACRFile = open(dirName+"/traj-" + str(n) + "/NACR.dat","w")
    NACTFile = open(dirName+"/traj-" + str(n) + "/NACT.dat","w")
    forceFile = open(dirName+"/traj-" + str(n) + "/force.dat","w")
    return activeStateFile, energyFileAd, energyFileDia, densityFiles, InitCondsFile, RFile, VFile, HelFile, probFile, randFile, NACRFile, NACTFile, forceFile

def closeFiles(activeStateFile, energyFileAd, energyFileDia, densityFiles, InitCondsFile, RFile, VFile, HelFile, probFile, randFile, NACRFile, NACTFile, forceFile):
    energyFileAd.close()    
    energyFileDia.close()    
    densityFiles[0].close()    
    densityFiles[1].close()    
    densityFiles[2].close()   
    densityFiles[3].close()   
    InitCondsFile.close()
    RFile.close()
    VFile.close()
    HelFile.close()
    probFile.close()
    randFile.close()
    NACTFile.close()
    NACRFile.close()
    forceFile.close()

def writeDensity(densityFiles,z_ad,active_state,activeStateFile,step,Uad):

    rho_ad = get_density_matrix( z_ad )
    rho_dia = get_diabatic_density( z_ad, active_state, Uad )

    outArrayRE = [ step * dtI, step * dtI / fs_to_au ]
    outArrayIM = [ step * dtI, step * dtI / fs_to_au ]

    # Write adiabatic density matrix
    elements = rho_ad.reshape(-1)
    for element in elements:
        outArrayRE.append( np.real( element ) )
        outArrayIM.append( np.imag( element ) )

    densityFiles[0].write( "\t".join(map("{:.5f}".format,outArrayRE)) + "\n")
    densityFiles[1].write( "\t".join(map("{:.5f}".format,outArrayIM)) + "\n")

    outArrayRE = [ step * dtI, step * dtI / fs_to_au ]
    outArrayIM = [ step * dtI, step * dtI / fs_to_au ]

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

def writeEnergy_dia(energyFileDia,z_ad,Uad,active_state,step,M,V,Hel):

    rho_dia = get_diabatic_density( z_ad, active_state, Uad )

    KIN = get_Kinetic( M, V )
    POT = 0
    for j in range( NStates ):
        POT += rho_dia[j,j] * Hel[j,j]
    TOT = KIN + POT

    energyFileDia.write( "{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\n".format(step * dtI, step * dtI / fs_to_au, KIN, POT, TOT) )

def writeR(R,RFile):
    RFile.write("\t".join(map("{:2.8}".format, R)) + "\n")

def writeV(V,VFile):
    VFile.write("\t".join(map("{:2.8}".format, V)) + "\n")

def writeHel(Hel,HelFile):
    outList = []
    for i in range(NStates):
        for j in range(NStates):
            outList.append( round(Hel[i,j],6) )
    HelFile.write( "\t".join(map(str,outList))+"\n" )

def writeNAC(deriv_coup, V, NACRFile, NACTFile, step):
    NACR = deriv_coup
    NACT = get_scalar_coupling( V, deriv_coup )


    outNACT    = [step*dtI]
    headerNACT = []
    for j in range( NStates ):
        for k in range( j+1, NStates ):
            headerNACT.append(f"{j}-{k}")
            outNACT.append( NACT[j,k] )
    if ( step == 0 ): NACTFile.write( "Time\t" + "\t".join(map(str,headerNACT)) + "\n")
    NACTFile.write( "\t".join(map("{:1.10f}".format, outNACT)) + "\n"  )

    outNACR    = [step*dtI]
    headerNACR = []
    for d in range( NDOF ):
        for j in range( NStates ):
            for k in range( j+1, NStates ):
                headerNACR.append(f"{j}-{k} (dof={d})")
                outNACR.append( NACR[j,k,d] )
    if ( step == 0 ): NACRFile.write( "Time\t" + "\t".join(map(str,headerNACR)) + "\n")
    NACRFile.write( "\t".join(map("{:1.10f}".format, outNACR)) + "\n"  )

def initCoeffs(InitCondsFile):# Initialization of the mapping Variables
    """
    Returns np.array((F)) of adiabatic electronic coefficients z_j
    """
    z = np.zeros( ( NStates ), dtype=complex )
    z[initState] = 1.0
    active_state = initState
    return z, active_state

def diabatic_to_adiabatic( MAT, Uad ):
    """
    Uad : Transformation matrix that diagonalizes the diabatic Hamiltonian
    MAT : Some diabatic matrix
    """
    #return Uad.T @ MAT @ Uad
    return np.einsum( "JK,KL...,LM->JM...", Uad.T, MAT, Uad) # FORWARD
    ###WRONG#####return np.einsum( "JK,KL...,LM->JM...", Uad, MAT, Uad.T) # BACKWARD

def adiabatic_to_diabatic( MAT, Uad ):
    """
    Uad : Transformation matrix that diagonalizes the diabatic Hamiltonian
    MAT : Some abatic matrix
    """
    #return Uad @ MAT @ Uad.T
    return np.einsum( "JK,KL...,LM->JM...", Uad, MAT, Uad.T) # FORWARD
    ###WRONG####return np.einsum( "JK,KL...,LM->JM...", Uad.T, MAT, Uad) # BACKWARD 

def get_diabatic_density( z_ad, active_state, Uad ):

    rho_dia = np.zeros((NStates,NStates),dtype=complex)
    rho_ad = np.zeros((NStates,NStates),dtype=complex)

    if ( Diabatic_Density_Type == 0 ):
        """
        \rho_jk = zz* (j !+ k)
        """
        rho_ad = get_density_matrix(z_ad)
        rho_dia = adiabatic_to_diabatic( rho_ad, Uad )

    if ( Diabatic_Density_Type == 1 ):
        """
        \rho_jk = 1.0 * (j == k and AS == j)
        """
        rho_ad[active_state,active_state] = 1.0 + 0.0j
        rho_dia = adiabatic_to_diabatic( rho_ad, Uad )

    if ( Diabatic_Density_Type == 2 ):
        """
        \rho_jk = {{AS,zz*},{z*z,AS}} = 1.0 * (j == k and AS == j) + zz* (j !+ k)
        """
        for j in range( NStates ):
            for k in range( j+1, NStates ):
                rho_ad[j,k] = z_ad[j] * np.conjugate( z_ad[k] )
                rho_ad[k,j] = np.conjugate( rho_ad[j,k] )
        rho_ad[active_state,active_state] = 1.0 + 0.0j
        rho_dia = adiabatic_to_diabatic( rho_ad, Uad )

    return rho_dia

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

def prop_el_coeffs_RK4(z_ad, Ead, deriv_coup, V, dtE, ESteps):
    """
    Propagates electornic coefficients
    """

    NACT = get_scalar_coupling(V, deriv_coup)
    E_mat = np.diag(Ead)

    def f( y ):
        return -1j * E_mat @ y - NACT @ y

    y0 = z_ad.copy()
    yt = y0.copy()

    for step in range( ESteps ):

        k1 = f(yt)
        k2 = f(yt + k1*dtE/2)
        k3 = f(yt + k2*dtE/2)
        k4 = f(yt + k3*dtE)

        yt += 1/6 * ( k1 + 2*k2 + 2*k3 + k4 ) * dtE


    z_ad = yt
    return z_ad

def prop_el_coeffs_RK_SCIPY(z_ad, Ead, deriv_coup, V, dtE, ESteps):
    
    def ODE( t, z_ad, E_mat, NACT ):
        return -1j * E_mat @ z_ad - NACT @ z_ad

    NACT = get_scalar_coupling(V, deriv_coup)
    E_mat = np.diag(Ead)

    z_ad_0 = np.copy(z_ad)
    sol = solve_ivp(ODE, t_span=[0,ESteps*dtE/2], y0=z_ad_0, t_eval=[ESteps*dtE/2], max_step=dtE, method="RK45", args=(E_mat, NACT))

    z_ad = sol.y[:,-1]

    return z_ad

def Force(dHel_ad, R, z, active_state, step, dHel0, forceFile, writeForce=True):
    """
    F = F0 (State-Ind.) + Fad (Hell.-Feyn.)
    Force only contributes from the active ADIABATIC state
    """
    F = np.zeros((len(R)))
    F -= dHel_ad[active_state,active_state,:] + dHel0[:]

    if ( writeForce == True ):
        forceFile.write( f"{step*dtI}\t" + "\t".join(map("{:1.8f}".format, F)) + "\n" )

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
        deriv_coup[:,:,d] = diabatic_to_adiabatic( dHel0 + dHel[:,:,d], Uad ) / Ediffs
    
    deriv_coup[np.diag_indices(len(deriv_coup))] = 0.0 # Replace diagonal with "0"

    return deriv_coup 

def get_scalar_coupling( V, deriv_coup ):
    return np.einsum( "d,ijd->ij", V, np.real(deriv_coup) )

def get_hop_prob( z_ad, z_ad_old, V, deriv_coup, active_state, dtI, probFile ):
    
    rho     = get_density_matrix( z_ad )
    POP_OLD = np.real( z_ad_old * np.conjugate(z_ad_old) )
    POP     = np.real( z_ad * np.conjugate(z_ad) )
    NACT    = get_scalar_coupling( V, deriv_coup )
    probs   = np.zeros(( NStates ))


    # No hopping if active state is gaining population
    if ( AS_POP_INCREASE == 1 and POP[active_state] > POP_OLD[active_state] ):
        probs[active_state] = 1.000
        return  probs

    for j in range( NStates ):
        if ( j != active_state ):
            b = -2 * np.real(rho[j,active_state]) * NACT[j,active_state]
            probs[j] = b * dtI / POP[active_state]

            if ( probs[j] < 0.0 ):
                probs[j] = 0.0

    #probs[active_state] = 1 - np.sum(probs) # DEPING SUGGEST
    probs[ probs < 0.0 ] = 0.0

    probFile.write( "\t".join(map("{:1.10f}".format,probs)) + "\n" )

    return probs

def hop_check( active_state, probs, randFile ):

    #############print("DEBUGGING: Setting random number to 0.1 ")
    rand = random()
    #rand = 0.1000000
    randFile.write( "\t".join(map("{:1.10f}".format,[rand])) + "\n" )

    hop_data = [False, active_state, active_state]
    
    sum_prob = np.zeros(( NStates+1 ))
    for j in range( 1, NStates+1 ): # 1,2,3...,NStates
        sum_prob[j] += np.sum( probs[:j] )
    
    for j in range( NStates ):
        #print( "State, SUM_PROB:", j, sum_prob[j], rand, sum_prob[j+1] )
        if ( sum_prob[j] < rand and rand <= sum_prob[j+1] ):
            if ( active_state != j ):
                print(f"HOP: {active_state} --> {j}")
                hop_data[0] = True
                hop_data[1] = active_state
                hop_data[2] = j
    
    return hop_data

def get_Kinetic( M, V ):
    return 0.5000 * np.einsum( "d,d,d->", M, V, V)

def get_Potential( Ead, active_state ):
    return Ead[active_state]

def get_Potential_Diff( Ead, hop_data ):
    start = hop_data[1]
    end = hop_data[2]
    return Ead[end] - Ead[start]

def get_decoherence( z_ad, active_state, Ead, KIN ):
    
    if ( decoherece_type == "IDC" ):
        z_ad = np.zeros((NStates), dtype=complex)
        z_ad[active_state] = 1.0 + 0.0j

    elif ( decoherece_type == "EDC" ):
        rho_update = 0.0 + 0.0j
        for j in range( NStates ):
            if ( j == active_state ): 
                exp_tau[j] = 1.
            else:
                PDIFF = np.abs( get_Potential_Diff( Ead, [True, active_state, j] ) )
                tau   = (1 + EDC_PARAM / KIN) / PDIFF
                z_ad[j] *= np.exp( - dtI / tau )
                rho_update += np.abs( z_ad[j] ) ** 2
        z_ad[active_state] *= np.sqrt( (1-rho_update)/np.abs(z_ad[active_state])**2 )

        """
        # Get new coefficients
        rho_update = 1
        for j in range( NStates ):
            z_ad *= exp_tau[j]
            rho_update -= z_ad[j] * np.conjugate(z_ad[j])
        
        rho_AA = z_ad[active_state] * np.conjugate(z_ad[active_state])
        z_ad[ active_state ] *= np.sqrt( rho_update / rho_AA )
        """

    return z_ad 

def evaluate_hop( active_state, hop_data, M, V, Ead, deriv_coup, z_ad ):

    KIN = get_Kinetic( M, V )
    POT = get_Potential( Ead, active_state )

    if( decoherece_type == "EDC" ):
        get_decoherence( z_ad, active_state, Ead, KIN )

    if ( hop_data[0] == True ):

        PDIFF = get_Potential_Diff( Ead, hop_data )

        if( decoherece_type == "IDC" ):
            get_decoherence( z_ad, active_state, Ead, KIN ) # Frustrated hops get decoherence as well

        start_state = hop_data[1]
        end_state = hop_data[2]

       
        det = 1
        if( rescale_type == "velocity" ): # NOT TESTED TODO
            print("VELOCITY RESCALING NOT YET TESTED.")
            a = 0.50000 * np.sum (M[:] * deriv_coup[start_state,end_state,:] ** 2)
            b = 1.00000 * np.sum( M[:] * deriv_coup[start_state,end_state,:] * V[:] )
            c = 1.00000 * PDIFF            
            det = b ** 2. - 4. * a * c

        elif( rescale_type == "momentum" ):
            a = 0.50000 * np.sum ( deriv_coup[start_state,end_state,:] ** 2 / M[:] )
            b = 1.00000 * np.sum( deriv_coup[start_state,end_state,:] * V[:] )
            c = 1.00000 * PDIFF         
            det = b ** 2. - 4. * a * c
           
        if ( rescale_type == 'energy' or det < 0 ):
            if ( end_state > start_state and KIN < np.abs(PDIFF) ):
                # Rejecting hop
                print("REJECTING HOP BASED ON LACK OF K.E.:")
                return V, z_ad, active_state

            # Calculate the uniform scaling factor based on energy criterion
            scale_factor = np.sqrt( 1 - PDIFF / KIN )
            V *= scale_factor

        else:
            if ( b < 0 ):
                scale_factor = 0.5 * ( b + np.sqrt(det)) / a
            else:
                scale_factor = 0.5 * ( b - np.sqrt(det)) / a

            if ( rescale_type == "velocity" ):
                V[:] -= scale_factor * deriv_coup[start_state,end_state,:]
            elif ( rescale_type == "momentum" ):
                V[:] -= scale_factor * deriv_coup[start_state,end_state,:] / M[:]


        # DEPING SUGGESTS TO TURN OFF STATE SWAPPING
        # Rearrange states based on hop
        if ( SWAP_COEFFS_HOP == 1 ):
            tmp = np.copy( z_ad[start_state] )
            z_ad[start_state] = z_ad[end_state]
            z_ad[end_state] = tmp

        active_state = end_state


    return M*V, z_ad, active_state

def check_phase( Uad, Uad_old, Ead ):

    V = np.zeros(( NStates, NStates ))
    P = np.zeros(( NStates, NStates ))

    V = Uad.T @ Uad
    for j in range( NStates ):
        for k in range( NStates ):
            E_diff = abs(Ead[j] - Ead[k])
            #print( V[j,k], E_diff )
            if ( E_diff < 1e-6 ):
                P[j,k] = V[j,k]

    if ( not np.allclose(P, np.identity( NStates ) ) ):
        print("PHASE CORRECTION:")
        print( P - np.identity( NStates ) )
        
    Uad = Uad @ P # Phase-corrected matrix

    return Uad

def VelVerF(R, P, z_ad, Uad_old, active_state, RFile, VFile, HelFile, energyFileAd, energyFileDia, densityFiles, activeStateFile, probFile, randFile, NACRFile, NACTFile, forceFile, step): # Ionic position, ionic momentum, etc.
    
    Hel = model.Hel(R) # Electronic Structure
    Ead, Uad = np.linalg.eigh(Hel) # Return adiabatic states and transformation matrix
    Uad = check_phase( Uad, Uad_old, Ead )
    Uad_old = Uad.copy()
    dHel = model.dHel(R) # Diabatic forces
    dHel0 = model.dHel0(R) # This is scalar. No need to transform ? But identity matrix will transform ?
    dHel_ad = diabatic_to_adiabatic( dHel, Uad ) # Transform forces to adiabatic representation
    deriv_coup = get_derv_coupling( dHel0, dHel, Uad, Ead )

    if ( step % NSkip == 0 ): 
        writeDensity(densityFiles,z_ad,active_state,activeStateFile,step,Uad)
        writeEnergy_ad(energyFileAd,active_state,step,M,P/M,Ead)
        writeEnergy_dia(energyFileDia,z_ad,Uad,active_state,step,M,P/M,Hel)
        writeR(R,RFile)
        writeR(P/M,VFile)
        writeNAC(deriv_coup, P/M, NACRFile, NACTFile, step)

    z_ad_old = z_ad.copy()


    ESteps = int(dtI/dtE)
    #z_ad = prop_el_coeffs_VV(z_ad, Ead, deriv_coup, P/M, dtE, ESteps//2)
    z_ad = prop_el_coeffs_RK4(z_ad, Ead, deriv_coup, P/M, dtE, ESteps//2)
    #z_ad = prop_el_coeffs_RK_SCIPY(z_ad, Ead, deriv_coup, P/M, dtE, ESteps//2)

    F1 = Force(dHel_ad, R, z_ad, active_state, step, dHel0, forceFile, writeForce=True)

    P += 0.5000 * F1 * dtI # Half-step velocity

    R += P / M * dtI # Full Step Position

    Hel = model.Hel(R) # Electronic Structure
    Ead, Uad = np.linalg.eigh(Hel) # Return adiabatic states and transformation matrices
    Uad = check_phase( Uad, Uad_old, Ead )
    dHel = model.dHel(R) # Diabatic forces
    dHel0 = model.dHel0(R) # This is scalar. No need to transform ?
    dHel_ad = diabatic_to_adiabatic( dHel, Uad ) # Transform forces to adiabatic representation
    deriv_coup = get_derv_coupling( dHel0, dHel, Uad, Ead )

    F2 = Force(dHel_ad, R, z_ad, active_state, step, dHel0, forceFile, writeForce=False)
    
    P += 0.5000 * F2 * dtI # Half-step Velocity

    #z_ad = prop_el_coeffs_VV(z_ad, Ead, deriv_coup, P/M, dtE, ESteps//2)
    z_ad = prop_el_coeffs_RK4(z_ad, Ead, deriv_coup, P/M, dtE, ESteps//2)
    #z_ad = prop_el_coeffs_RK_SCIPY(z_ad, Ead, deriv_coup, P/M, dtE, ESteps//2)


    ##### START FSSH Hopping #####
    probs = get_hop_prob(z_ad, z_ad_old, P/M, deriv_coup, active_state, dtI, probFile)
    hop_data = hop_check(active_state, probs, randFile)
    P, z_ad, active_state = evaluate_hop(active_state, hop_data, M, P/M, Ead, deriv_coup, z_ad)
    ##### END FSSH Hopping #####

    return R, P, z_ad, active_state, Uad

def RunIterations(n): # This is parallelized already. "Main" for each trajectory.

    cleanDir(n)
    activeStateFile, energyFileAd, energyFileDia, densityFiles, InitCondsFile, RFile, VFile, HelFile, probFile, randFile, NACRFile, NACTFile, forceFile = initFiles(n) # Makes file objects

    R,P = model.initR() # Initialize nuclear DOF

    z, active_state = initCoeffs(InitCondsFile)
    Uad = np.identity( NStates )

    for step in range(NSteps):
        #print ("Step:", step, active_state)
        R, P, z, active_state, Uad = VelVerF(R, P, z, Uad, active_state, RFile, VFile, HelFile, energyFileAd, energyFileDia, densityFiles, activeStateFile, probFile, randFile, NACRFile, NACTFile, forceFile, step)
    
    closeFiles(activeStateFile, energyFileAd, energyFileDia, densityFiles, InitCondsFile, RFile, VFile, HelFile, probFile, randFile, NACRFile, NACTFile, forceFile)

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





