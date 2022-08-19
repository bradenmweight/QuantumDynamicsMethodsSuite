import numpy as np
#from matplotlib import pyplot as plt
import math

NTraj = 10 ** 4

time               = np.loadtxt("traj-0/active_state.dat")[:,[0,1]]
NStates            = len(np.loadtxt("traj-0/active_state.dat")[0,2:])
NSteps             = len( time )
coeffs_pop_ad      = np.zeros(( NSteps, NStates ))
coeffs_pop_dia_0   = np.zeros(( NSteps, NStates ))
coeffs_pop_dia_1   = np.zeros(( NSteps, NStates ))
coeffs_pop_dia_2   = np.zeros(( NSteps, NStates ))
active_state       = np.zeros(( NSteps, NStates ))


for traj in range( NTraj ):
    print( f"traj {traj} of {NTraj}" )
    try:
        active_state     += np.loadtxt(f"traj-{traj}/active_state.dat")[:,2:]
    except OSError:
        NTraj = traj
        break
    if ( NStates == 2 ):
        coeffs_pop_ad    += np.loadtxt(f"traj-{traj}/density_ad_re.dat")[:,[2,5]]
        coeffs_pop_dia_0   += np.loadtxt(f"traj-{traj}/density_dia_re_0.dat")[:,[2,5]]        
        coeffs_pop_dia_1   += np.loadtxt(f"traj-{traj}/density_dia_re_1.dat")[:,[2,5]]        
        coeffs_pop_dia_2   += np.loadtxt(f"traj-{traj}/density_dia_re_2.dat")[:,[2,5]]        
    elif ( NStates == 3 ):
        coeffs_pop_ad    += np.loadtxt(f"traj-{traj}/density_ad_re.dat")[:,[2,6,10]]
        coeffs_pop_dia_0   += np.loadtxt(f"traj-{traj}/density_dia_re_0.dat")[:,[2,6,10]]
        coeffs_pop_dia_1   += np.loadtxt(f"traj-{traj}/density_dia_re_1.dat")[:,[2,6,10]]
        coeffs_pop_dia_2   += np.loadtxt(f"traj-{traj}/density_dia_re_2.dat")[:,[2,6,10]]

print(f"There were {NTraj} of {NTraj} good trajectories.")
active_state    /= NTraj
coeffs_pop_ad   /= NTraj
coeffs_pop_dia_0  /= NTraj
coeffs_pop_dia_1  /= NTraj
coeffs_pop_dia_2  /= NTraj


if ( NStates == 2 ):
    np.savetxt("POP_AS.dat", np.c_[ time[:,0], time[:,1], active_state[:,0], active_state[:,1] ], fmt="%1.5f")
    np.savetxt("POP_COEFF_AD.dat", np.c_[ time[:,0], time[:,1], coeffs_pop_ad[:,0], coeffs_pop_ad[:,1] ], fmt="%1.5f")
    np.savetxt("POP_COEFF_DIA_0.dat", np.c_[ time[:,0], time[:,1], coeffs_pop_dia_0[:,0], coeffs_pop_dia_0[:,1] ], fmt="%1.5f")
    np.savetxt("POP_COEFF_DIA_1.dat", np.c_[ time[:,0], time[:,1], coeffs_pop_dia_1[:,0], coeffs_pop_dia_1[:,1] ], fmt="%1.5f")
    np.savetxt("POP_COEFF_DIA_2.dat", np.c_[ time[:,0], time[:,1], coeffs_pop_dia_2[:,0], coeffs_pop_dia_2[:,1] ], fmt="%1.5f")
       
elif ( NStates == 3 ):
    np.savetxt("POP_AS.dat", np.c_[ time[:,0], time[:,1], active_state[:,0], active_state[:,1], active_state[:,2] ], fmt="%1.5f")
    np.savetxt("POP_COEFF_AD.dat", np.c_[ time[:,0], time[:,1], coeffs_pop_ad[:,0], coeffs_pop_ad[:,1], coeffs_pop_ad[:,2] ], fmt="%1.5f")
    np.savetxt("POP_COEFF_DIA_0.dat", np.c_[ time[:,0], time[:,1], coeffs_pop_dia_0[:,0], coeffs_pop_dia_0[:,1], coeffs_pop_dia_0[:,2] ], fmt="%1.5f")
    np.savetxt("POP_COEFF_DIA_1.dat", np.c_[ time[:,0], time[:,1], coeffs_pop_dia_1[:,0], coeffs_pop_dia_1[:,1], coeffs_pop_dia_1[:,2] ], fmt="%1.5f")
    np.savetxt("POP_COEFF_DIA_2.dat", np.c_[ time[:,0], time[:,1], coeffs_pop_dia_2[:,0], coeffs_pop_dia_2[:,1], coeffs_pop_dia_2[:,2] ], fmt="%1.5f")










