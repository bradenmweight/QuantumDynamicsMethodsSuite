import numpy as np
from matplotlib import pyplot as plt
import subprocess as sp
import os
import multiprocessing as mp

def get_globals():
    global NStates, NSteps, NTraj, dtI, NCPUS
    global do_Partial_Analysis, do_Pop_Only, do_Parallel_Read
    global typeDEN, COLUMN
    global time, wF, wB
    global OUTER_DIR, IMAGE_DIR

    NStates = 7    # Number of Electronic States
    NSteps  = 1000 # Number of "SAVED" Nuclear Time Steps -- NStepsReal / NSkip
    NTraj   = 10 # Number of Trajectories
    dtI     = 41.341/2*10 # 1.0 # a.u. # (Nuclear Step, dtI) * NSkip , Only used for FFT in abs. spec.


    do_Partial_Analysis = False
    do_Pop_Only = True
    do_Parallel_Read = True
    NCPUS = 12

    typeDEN = "ALL" # "ALL", "UPT", "DIAG", "COLUMN"
    COLUMN = 0 # Choose initial state. Recall 0 = G.S.


    time = np.zeros(( NSteps, 2 )) # Convert to fs
    wF = np.zeros((NStates,NStates,NTraj,NSteps,NStates,NStates),dtype=complex) # Forward Mapping Kernel
    wB = np.zeros((NStates,NStates,NTraj,NSteps,NStates,NStates),dtype=complex) # Backward Mapping Kernel

    print(f"\tMemory size of one numpy array: ({round(wF.size * wF.itemsize * 10 ** -6,2)} MB,{round(wF.size * wF.itemsize * 10 ** -9,2)} GB)" )


    OUTER_DIR = "TRAJ_spin-PLDM"
    IMAGE_DIR = f"{OUTER_DIR}/data_images_correlations/"
    if ( not os.path.isdir(OUTER_DIR) ):
        print( f"{OUTER_DIR} not found. Run above 'OUTER_DIR'." )
        exit()
    sp.call(f"mkdir -p {IMAGE_DIR}",shell=True)

    return wF, wB


def read_kernels( typeDEN="ALL" ):

    if ( typeDEN == "ALL" ):
        readSTATES = [ (n,m) for n in range(NStates) for m in range(NStates) ]
    elif( typeDEN == "DIAG" ):
        readSTATES = [ (n,n) for n in range(NStates) ]
    elif( typeDEN == "UPT" ):
        readSTATES = [ (n,m) for n in range(NStates) for m in range(n,NStates) ]

    for n,m in readSTATES :
        print(f"Working on element = ({n}, {m})")
        for traj in range( NTraj ):

            ### READ IN MAPPING KERNELS FOR ALL TRAJECTORIES AND INITIALLY FOCUSED STATES ###
            #print(f"Working on TRAJ = {traj}")
            tmpF = np.loadtxt( f"{OUTER_DIR}/Partial__{n}{m}/traj-{traj}/mapping_F.dat", dtype=complex )
            tmpB = np.loadtxt( f"{OUTER_DIR}/Partial__{n}{m}/traj-{traj}/mapping_B.dat", dtype=complex )
            #print( tmpF[0,:] )
            count = 0
            for j in range( NStates ):
                for k in range( NStates ):
                    wF[n,m,traj,:,j,k] = tmpF[:NSteps, 2+count]
                    wB[n,m,traj,:,j,k] = tmpB[:NSteps, 2+count]
                    count += 1

            if ( traj == 0 ): time = tmpF[:NSteps,:2]

    return wF, wB

def read_kernel_parallel( nm ):

    n,m = nm

    wFnm = np.zeros((NTraj,NSteps,NStates,NStates),dtype=complex)
    wBnm = np.zeros((NTraj,NSteps,NStates,NStates),dtype=complex)

    print(f"Working on element = ({n}, {m})")
    for traj in range( NTraj ):

        ### READ IN MAPPING KERNELS FOR ALL TRAJECTORIES AND INITIALLY FOCUSED STATES ###
        #print(f"Working on TRAJ = {traj}")
        tmpF = np.loadtxt( f"{OUTER_DIR}/Partial__{n}{m}/traj-{traj}/mapping_F.dat", dtype=complex )
        tmpB = np.loadtxt( f"{OUTER_DIR}/Partial__{n}{m}/traj-{traj}/mapping_B.dat", dtype=complex )
        #print( tmpF[0,:] )
        count = 0
        for j in range( NStates ):
            for k in range( NStates ):
                wFnm[traj,:,j,k] = tmpF[:NSteps, 2+count]
                wBnm[traj,:,j,k] = tmpB[:NSteps, 2+count]
                count += 1

    return wFnm, wBnm

def get_time():

    t = np.loadtxt( f"{OUTER_DIR}/Partial__00/traj-0/mapping_F.dat", dtype=complex )[:NSteps,:2]

    return np.real( t )

def get_CAB( wF, wB, A, B, return_Partials = False ):

    # Get matrix multiplication for correlation function
    AwBw = np.einsum( "ab,NMTtbc,cd,NMTtde->NMTtae",  A, wB, B, wF )    

    # Compute trace in correlation function
    TrAwBw = np.einsum( "NMTtaa->NMTt", AwBw[:,:,:,:,:,:] )

    ### AVERAGE OVER EACH TRAJECTORY AT EACH TIME STEP ###
    TrAwBw_ave = np.average( TrAwBw[:,:,:,:],axis=2 )

    ### ADD ALL FOCUSED INITIAL CONDITIONS ###         
    CAB = np.einsum( "NMt->t", TrAwBw_ave[:,:,:] )

    if ( return_Partials ):
        return CAB, TrAwBw_ave
    else:
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

def get_POPULATION_from_initPOP( wF, wB, init = 1, track = 1 ): # Track which element


    # Get (1) part of correlation
    rho0 = np.zeros(( NStates,NStates ), dtype=complex)
    A = np.zeros(( NStates,NStates ), dtype=complex)
    B = np.zeros(( NStates,NStates ), dtype=complex)

    A[init,init] = 1.0
    B[track,track] = 1.0

    print( "Population: A =\n", np.real(A)   )
    print( "Population: B =\n", np.real( B ) )

    CAB = get_CAB( wF, wB, A, B )

    np.savetxt(f"{IMAGE_DIR}/P_init{init}_track{track}_NTraj{NTraj}.dat", np.c_[time, np.real(CAB) ] )

    return np.real(CAB)

def get_COHERENCE_FROM_initPOP( wF, wB, init = 1, track = [1,1], return_Partials=False ): # Track which element


    rho0 = np.zeros(( NStates,NStates ), dtype=complex)
    A = np.zeros(( NStates,NStates ), dtype=complex)
    B = np.zeros(( NStates,NStates ), dtype=complex)

    A[init,init] = 1.0
    B[track[0],track[1]] = 1.0

    print( "Population: A =\n", np.real(A) )
    print( "Coherence:  B =\n", np.real(B) )

    if ( return_Partials ):
        CAB, TrAwBw_ave = get_CAB( wF, wB, A, B, return_Partials=True )
        for j in range( NStates ):
            for k in range( NStates ):
                np.savetxt(f"{IMAGE_DIR}/P_init{init}_track{track[0]}{track[1]}_NTraj{NTraj}_P{j}{k}.dat", np.c_[np.real(TrAwBw_ave[j,k]), np.imag(TrAwBw_ave[j,k]) ] )

    else:
        CAB = get_CAB( wF, wB, A, B )

    np.savetxt(f"{IMAGE_DIR}/P_init{init}_track{track[0]}{track[1]}_NTraj{NTraj}.dat", np.c_[np.real(CAB), np.imag(CAB) ] )

    if ( return_Partials ):
        return CAB, TrAwBw_ave
    else:
        return CAB


################# START MAIN PROGRAM #################

def main():

    wF, wB = get_globals()

    if ( do_Parallel_Read == True ):
        with mp.Pool(processes=NCPUS) as pool:
            if ( typeDEN == "ALL" ):
                nms = [ [n,m] for n in range( NStates ) for m in range( NStates ) ] 
                tmp = pool.map(read_kernel_parallel,nms)
                for count,(n,m) in enumerate(nms):
                    wF[n,m] = np.array( tmp[count][0] )
                    wB[n,m] = np.array( tmp[count][1] )
            elif( typeDEN == "UPT" ):
                print("Approximating inital conditions to upper triangle. Assumed Hermitian.")
                nms = [ [n,m] for n in range( NStates ) for m in range( n, NStates ) ] 
                tmp = pool.map(read_kernel_parallel,nms)
                for count,(n,m) in enumerate(nms):
                    wF[n,m] = np.array( tmp[count][0] )
                    wF[m,n] = np.conjugate( np.array( tmp[count][0] ) )
                    wB[n,m] = np.array( tmp[count][1] )
                    wB[m,n] = np.conjugate( np.array( tmp[count][1] ) )
            elif( typeDEN == "DIAG" ):
                print("!!! WARNING !!! Approximating inital conditions to diagonal. Coherences are zeros. Probably bad.")
                nms = [ [n,n] for n in range( NStates ) ] 
                tmp = pool.map(read_kernel_parallel,nms)
                for count,(n,m) in enumerate(nms):
                    wF[n,m] = np.array( tmp[count][0] )
                    wB[n,m] = np.array( tmp[count][1] )
            elif( typeDEN == "COLUMN" ):
                print(f"!!! WARNING !!! Approximating inital conditions to single column. All else is zero. Choose column to be initial state. Column = {column}. Assuming Hermitian.")
                nms = [ [column,n] for n in range( NStates ) ] 
                tmp = pool.map(read_kernel_parallel,nms)
                for count,(n,m) in enumerate(nms):
                    wF[n,m] = np.array( tmp[count][0] )
                    wF[m,n] = np.conjugate( np.array( tmp[count][0] ) )
                    wB[n,m] = np.array( tmp[count][1] )
                    wB[m,n] = np.conjugate( np.array( tmp[count][1] ) )

    else:
        wF, wB = read_kernels()

    time = get_time()
    rho = np.zeros(( NStates, NStates, NSteps ), dtype=complex)

    if ( do_Pop_Only == True ):
        # Plot all population elements
        for state_j in range( NStates ):
            rho[state_j,state_j,:] = get_COHERENCE_FROM_initPOP( wF, wB, init = 0, track = [state_j,state_j], return_Partials=False ) # Choose state to intialize and track
            plt.plot( time[:,0], np.real(rho[state_j,state_j]), linewidth=3, alpha=0.6, label=f"P{state_j+1}" )

    else:
        rho_partial = np.zeros(( NStates, NStates, NStates, NStates, NSteps ), dtype=complex)
        # Plot all density matrix elements
        for state_j in range( NStates ):
            for state_k in range( NStates ):
                rho[state_j,state_k,:], rho_partial[:,:,state_j,state_k,:] = get_COHERENCE_FROM_initPOP( wF, wB, init = 1, track = [state_j,state_k], return_Partials=True ) # Choose state to intialize and track
                if ( state_j == state_k ):
                    plt.plot( time[:,0], np.real(rho[state_j,state_k]), linewidth=3, alpha=0.6, label=r"$\rho$"+f"{state_j}{state_k}" )
                else:
                    plt.plot( time[:,0], np.real(rho[state_j,state_k]),linewidth=3, alpha=0.6, label=r"$\rho$"+f"{state_j}{state_k} (RE)" )
                    plt.plot( time[:,0], np.imag(rho[state_j,state_k]),linewidth=3, alpha=0.6, label=r"$\rho$"+f"{state_j}{state_k} (IM)" )





    plt.legend()
    plt.xlim(0)
    plt.ylim(-0.4,1.05)
    plt.xlabel("Time (a.u.)",fontsize=15)
    plt.ylabel("Density Matrix Elements",fontsize=15)
    plt.title(f"Tully #1 (NTraj = {NTraj})",fontsize=15)
    plt.tight_layout()
    plt.savefig( f"{IMAGE_DIR}/P_init1_trackALL_NTraj{NTraj}_TypeDEN{typeDEN}.jpg", dpi=400 )
    #plt.clf()

    if ( do_Partial_Analysis ):

        # Plot decomposed partial contributions
        for P1 in range( NStates ): 
            for P2 in range( NStates ): 
                for state_j in range( NStates ):
                    for state_k in range( state_j,NStates ):
                        if ( state_j == state_k ):
                            tmp = np.real(rho_partial[P1,P2,state_j,state_k,:])
                            plt.plot( time[:,0], tmp, "--",linewidth=2, label=f"{state_j}{state_k} (P{P1}{P2})" )
                        else:
                            tmp = np.real(rho_partial[P1,P2,state_j,state_k,:])
                            plt.plot( time[:,0], tmp, "--",linewidth=2, label=f"{state_j}{state_k} (P{P1}{P2}) (RE)" )
                            tmp = np.imag(rho_partial[P1,P2,state_j,state_k,:])
                            plt.plot( time[:,0], tmp, "--",linewidth=2, label=f"{state_j}{state_k} (P{P1}{P2}) (IM)" )
        
        plt.legend()
        plt.xlim(0)
        plt.ylim(-0.4,1.05)
        plt.xlabel("Time (a.u.)",fontsize=15)
        plt.ylabel("Density Matrix Elements",fontsize=15)
        plt.title(f"Tully #1 (NTraj = {NTraj})",fontsize=15)
        plt.tight_layout()
        plt.savefig( f"{IMAGE_DIR}/P_init1_trackALL_NTraj{NTraj}_TypeDEN{typeDEN}_PjkDecomposed.jpg", dpi=400 )
        plt.clf()



if ( __name__ == "__main__" ):
    main()


















