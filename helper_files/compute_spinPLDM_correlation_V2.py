import numpy as np
from matplotlib import pyplot as plt
import subprocess as sp
import os, sys
import multiprocessing as mp

def get_globals():
    global NStates, NSteps, NTraj, dtI, NCPUS, initState
    global do_Partial_Analysis, do_Pop_Only, do_Parallel
    global typeDEN, COLUMN
    global time, wF, wB
    global OUTER_DIR, IMAGE_DIR

    NStates   = 7    # Number of Electronic States
    initState = 0 # 0,1,2,3,...
    NSteps    = 2000 # Number of "SAVED" Nuclear Time Steps -- NStepsReal / NSkip
    NTraj     = int( sys.argv[1] ) # Number of Trajectories
    dtI       = 41.341/2*10 # 1.0 # a.u. # (Nuclear Step, dtI) * NSkip , Only used for FFT in abs. spec.


    do_Partial_Analysis = False
    do_Pop_Only = True
    do_Parallel = True
    NCPUS = 200

    typeDEN = "ALL" # "ALL", "UPT", "DIAG", "COLUMN"
    COLUMN = initState # Choose initial state. Recall 0 = G.S. -- TODO


    time = np.zeros(( NSteps, 2 )) # Convert to fs

    OUTER_DIR = "TRAJ_spin-PLDM_savePOP"
    IMAGE_DIR = f"{OUTER_DIR}/data_images_correlations/"
    if ( not os.path.isdir(OUTER_DIR) ):
        print( f"{OUTER_DIR} not found. Run above 'OUTER_DIR'." )
        exit()
    sp.call(f"mkdir -p {IMAGE_DIR}",shell=True)

    return NCPUS



def get_time():

    t = np.loadtxt( f"{OUTER_DIR}/Partial__00/traj-0/mapping_F.dat", dtype=complex )[:NSteps,:2]

    return np.real( t )

def get_Partial_Traj_Trace( A, B ): # We can parallelize this method. Most of memory is just single trajectory data. Could have N^2 parallel treatments.
    """
    Computes matrix multiplication and trace. Then save to file in TRAJ_/Partial_NM/
    """

    for N in range( NStates ):
        for M in range( NStates ):
            print( f"Partial trajectory trace for {N}{M}" )
            for traj in range( NTraj ):
                tmpF = np.loadtxt( f"{OUTER_DIR}/Partial__{N}{M}/traj-{traj}/mapping_F.dat", dtype=complex )
                tmpB = np.loadtxt( f"{OUTER_DIR}/Partial__{N}{M}/traj-{traj}/mapping_B.dat", dtype=complex ) # Already daggered
                count = 0
                wF = np.zeros(( NSteps, NStates, NStates ), dtype=complex)
                wB = np.zeros(( NSteps, NStates, NStates ), dtype=complex)
                for j in range( NStates ):
                    for k in range( NStates ):
                        wF[:,j,k] = tmpF[:NSteps, 2+count]
                        wB[:,j,k] = tmpB[:NSteps, 2+count]
                        count += 1

                AwBw = np.einsum( "ab,tbc,cd,tde->tae",  A, wB, B, wF )
                TrAwBw = np.einsum( "taa->t", AwBw )
                np.save( f"{OUTER_DIR}/Partial__{N}{M}/TrAwBw_{traj}_{NTraj}.npy", TrAwBw[:] )

def get_Partial_Average(): # We can parallelize thid method. Only need N^2 parallel treatments.
    """
    Computes trajectory average for each partial
    """
    for N in range( NStates ):
        for M in range( NStates ):
            print( f"Partial average for {N}{M}" )
            TrAwBw = np.zeros(( NTraj, NSteps ))
            for traj in range( NTraj ):
                print( f"Partial average for {N}{M}" )
                TrAwBw[traj,:] = np.load( f"{OUTER_DIR}/Partial__{N}{M}/TrAwBw_{traj}_{NTraj}.npy" )
            TrAwBw_ave = np.average( TrAwBw[:,:], axis=0 )
            np.save( f"{OUTER_DIR}/Partial__{N}{M}/TrAwBw_AVERAGE_{NTraj}.npy", TrAwBw_ave[:] )

def get_Partial_Sum():
    """
    Computes sum for all partials
    """
    print( f"Computing sum of all partials" )
    CAB = np.zeros(( NSteps ), dtype=complex)
    for N in range( NStates ):
        for M in range( NStates ):
            CAB[:] += np.load( f"{OUTER_DIR}/Partial__{N}{M}/TrAwBw_AVERAGE_{NTraj}.npy" )

    return CAB



def get_Partial_Traj_Trace_parallel( ABnms ): # We can parallelize this method. Most of memory is just single trajectory data. Could have N^2 parallel treatments.
    """
    Computes matrix multiplication and trace. Then save to file in TRAJ_/Partial_NM/
    """
    A,B,traj,N,M = ABnms[0], ABnms[1], ABnms[2], ABnms[3], ABnms[4]
    #print( f"Partial trajectory trace for {N}{M} for traj {traj}" )
    tmpF = np.loadtxt( f"{OUTER_DIR}/Partial__{N}{M}/traj-{traj}/mapping_F.dat", dtype=complex )
    tmpB = np.loadtxt( f"{OUTER_DIR}/Partial__{N}{M}/traj-{traj}/mapping_B.dat", dtype=complex ) # Already daggered
    count = 0
    wF = np.zeros(( NSteps, NStates, NStates ), dtype=complex)
    wB = np.zeros(( NSteps, NStates, NStates ), dtype=complex)
    count = 0
    #wF = [ tmpF[:NSteps, 2+j+k] for j in range(NStates) for k in range(NStates) ]
    #wB = [ tmpB[:NSteps, 2+j+k] for j in range(NStates) for k in range(NStates) ]
    for j in range( NStates ):
        for k in range( NStates ):
            wF[:,j,k] = tmpF[:NSteps, 2+count]
            wB[:,j,k] = tmpB[:NSteps, 2+count]
            count += 1

    AwBw = np.einsum( "ab,tbc,cd,tde->tae",  A, wB, B, wF )
    TrAwBw = np.einsum( "taa->t", AwBw )
    np.save( f"{OUTER_DIR}/Partial__{N}{M}/traj-{traj}/TrAwBw_{traj}_{NTraj}.npy", TrAwBw[:] )

def get_Partial_Average_parallel( nms ): # We can parallelize thid method. Only need N^2 parallel treatments.
    """
    Computes trajectory average for each partial
    """
    N,M = nms[0], nms[1]
    #print( f"Partial average for {N}{M}" )
    TrAwBw = np.zeros(( NTraj, NSteps ), dtype=complex)
    for traj in range( NTraj ):
        TrAwBw[traj,:] = np.load( f"{OUTER_DIR}/Partial__{N}{M}/traj-{traj}/TrAwBw_{traj}_{NTraj}.npy" )
    TrAwBw_ave = np.average( TrAwBw[:,:], axis=0 )
    np.save( f"{OUTER_DIR}/Partial__{N}{M}/TrAwBw_AVERAGE_{NTraj}.npy", TrAwBw_ave[:] )

def get_CAB( A, B, NCPUS ):
    """
    This method is friendly to memory usage. Might take some time.
    """

    if ( do_Parallel == False ):
        get_Partial_Traj_Trace( A, B ) # Tr[ AwBw ] per trajectory
        get_Partial_Average() # Read Tr[AwBw] and average over trajectories
        CAB = get_Partial_Sum() # Size = NSteps
    else:
        with mp.Pool(processes=NCPUS) as pool:
            print(f"Running with {NStates} states and up to {NCPUS} cores.")
            ABTnms = [ [A,B,traj,n,m] for n in range( NStates ) for m in range( NStates ) for traj in range(NTraj) ] 
            pool.map(get_Partial_Traj_Trace_parallel, ABTnms)
            
        if( NCPUS > NStates * NStates ):
            NCPUS = NStates*NStates
        with mp.Pool(processes=NCPUS) as pool:
            nms = [ [n,m] for n in range( NStates ) for m in range( NStates ) ]
            pool.map(get_Partial_Average_parallel, nms)
        
        CAB = get_Partial_Sum() # Size = NSteps

    return CAB

def ABS_SPECTRA(rho0,A,B):
    # R1 = (1) - (2)
    # (1) = i Tr[ \mu(t) * ( \mu_{01}*10 + \mu_{02}*20 ) ]
    # (2) = i Tr[ \mu(t) * ( \mu_{01}*01 + \mu_{02}*02 ) ]
    # mu_{01} / mu_{02} = -5

    # Get (1) part of correlation
    #rho0 = np.zeros(( NStates,NStates ), dtype=complex)
    #A = np.zeros(( NStates,NStates ), dtype=complex)
    #B = np.zeros(( NStates,NStates ), dtype=complex)

    #rho0[0,0] = 1.0

    #A[0,1] = 5.0
    #A[0,2] = 1.0
    #A[2,0] = 1.0
    #A[1,0] = 5.0

    print( "(1) A =\n", np.real(rho0 @ A) )
    print( "(1) B =\n", np.real( B ) )
    #print( "(2) A =\t", np.real(A @ rho0) )

    #B[0,1] = 5.0
    #B[0,2] = 1.0
    #B[2,0] = 1.0
    #B[1,0] = 5.0
    CAB_R1 = get_CAB( rho0 @ A, B )

    # Get (2) part of correlation
    #rho0 = np.zeros(( NStates,NStates ), dtype=complex)
    #A = np.zeros(( NStates,NStates ), dtype=complex)
    #B = np.zeros(( NStates,NStates ), dtype=complex)

    #rho0[0,0] = 1.0

    #A[0,1] = 5.0
    #A[0,2] = 1.0
    #A[2,0] = 5.0
    #A[1,0] = 1.0

    print( "(1) A =\n", np.real( A @ rho0 ) )
    print( "(1) B =\n", np.real( B ) )

    #B[0,1] = 5.0
    #B[0,2] = 1.0
    #B[2,0] = 5.0
    #B[1,0] = 1.0
    CAB_R2 = get_CAB( A, B )

    R1 = 1j * ( CAB_R1 - CAB_R2 )

    plt.plot( np.arange(NSteps)*dtI, np.real(R1),    label="$\sum_{jk}  RE(C_{AB})_{jk}$" )
    plt.plot( np.arange(NSteps)*dtI, np.imag(R1),    label="$\sum_{jk}  IM(C_{AB})_{jk}$" )
    #plt.plot( np.arange(NSteps)*dtI, np.real(CAB_11[:]), label="$(C_{AB})_{11}$" )
    #plt.plot( np.arange(NSteps)*dtI, np.real(CAB_10[:]), label="$(C_{AB})_{10}$" )
    #plt.plot( np.arange(NSteps)*dtI, np.real(CAB_12[:]), label="$(C_{AB})_{12}$" )
    #plt.plot( np.arange(NSteps)*dtI, np.real(CAB_13[:]), label="$(C_{AB})_{13}$" )


    plt.legend()
    plt.xlabel("Time (a.u.)",fontsize=15)
    plt.ylabel("$C_{AB}$",fontsize=15)
    plt.tight_layout()
    plt.savefig( f"{IMAGE_DIR}/CAB.jpg" )
    plt.clf()


    ###### PERFORM FOURIER TRANSFORM OF C_{mu mu} FOR SPECTRA ######
    def get_FFT( f_t, dt):
        E = np.fft.fftfreq(len(f_t)) * (2.0 * np.pi / dt)
        #E = np.fft.fftfreq(5000) * (2.0 * np.pi / dt)
        f_E = np.fft.fft( f_t, norm="ortho" ) / np.sqrt(len(f_t))
        #f_E = np.fft.fft( f_t, norm="ortho", n=5000 ) / np.sqrt(len(f_t))
    return E, f_E

    #E, R1_E = get_FFT( np.imag(R1) , dtI ) # Definitely wrong
    E, R1_E = get_FFT( R1 , dtI )

    cminv2au = 4.55633*1e-6
    au2eV = 27.2114

    #E *= au2eV # Hartrees to eV
    E *= 1/cminv2au # Hartrees to cm^-1

    plt.plot( E, np.real(R1_E), label="Re[FFT[C]]" ) # 220000 cm^-1 / a.u.
    plt.plot( E, np.imag(R1_E), label="Im[FFT[C]]" ) # 220000 cm^-1 / a.u.
    plt.legend()
    #plt.xlim(-5,5)
    #plt.ylim(-0.2,0.2)
    plt.xlabel("Energy (eV)",fontsize=15)
    plt.ylabel("$C_{AB} (E)$",fontsize=15)
    plt.tight_layout()
    plt.savefig( f"{IMAGE_DIR}/CAB_SPECTRA.jpg" )
    plt.clf()



    np.savetxt(f"{IMAGE_DIR}/CAB_t.dat", R1 )
    np.savetxt(f"{IMAGE_DIR}/CAB_E.dat", np.c_[ E, np.real(R1_E), np.imag(R1_E) ] )  

def get_COHERENCE_FROM_initPOP( init = 1, track = [1,1], NCPUS=1 ):


    rho0 = np.zeros(( NStates,NStates ), dtype=complex)
    A = np.zeros(( NStates,NStates ), dtype=complex)
    B = np.zeros(( NStates,NStates ), dtype=complex)

    A[init,init] = 1.0
    B[track[0],track[1]] = 1.0

    print( "Population: A =\n", np.real(A) )
    print( "Coherence:  B =\n", np.real(B) )

    CAB = get_CAB( A, B, NCPUS )

    np.savetxt(f"{IMAGE_DIR}/P_init{init}_track{track[0]}{track[1]}_NTraj{NTraj}.dat", np.c_[np.real(CAB), np.imag(CAB) ] )

    return CAB


################# START MAIN PROGRAM #################

def main():

    NCPUS = get_globals()
    time = get_time()

    rho = np.zeros(( NStates, NStates, NSteps ), dtype=complex)

    # Plot all population elements
    for state_j in range( NStates ):
        rho[state_j,state_j,:] = get_COHERENCE_FROM_initPOP( init = initState, track = [state_j,state_j], NCPUS=NCPUS ) # Choose state to intialize and track
        plt.plot( time[:,0], np.real(rho[state_j,state_j]), linewidth=3, alpha=0.6, label=f"P{state_j+1}" )

    plt.legend()
    plt.xlim(0)
    plt.ylim(-0.4,1.05)
    plt.xlabel("Time (a.u.)",fontsize=15)
    plt.ylabel("Density Matrix Elements",fontsize=15)
    plt.title(f"NTraj = {NTraj}",fontsize=15)
    plt.tight_layout()
    plt.savefig( f"{IMAGE_DIR}/P_init{initState}_trackALL_NTraj{NTraj}_TypeDEN{typeDEN}.jpg", dpi=400 )
    plt.clf()


if ( __name__ == "__main__" ):
    main()


















