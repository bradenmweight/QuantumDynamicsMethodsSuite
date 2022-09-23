import numpy as np
#from matplotlib import pyplot as plt
import math

NTraj = 10 ** 5

time               = np.loadtxt("traj-0/active_state.dat")[:,[0,1]]
NStates            = len(np.loadtxt("traj-0/active_state.dat")[0,2:])
NSteps             = len( time )
coeffs_pop_ad      = np.zeros(( NSteps, NStates ))
coeffs_pop_dia     = np.zeros(( NSteps, NStates ))
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
        coeffs_pop_dia   += np.loadtxt(f"traj-{traj}/density_dia_re.dat")[:,[2,5]]            
    elif ( NStates == 3 ):
        coeffs_pop_ad    += np.loadtxt(f"traj-{traj}/density_ad_re.dat")[:,[2,6,10]]
        coeffs_pop_dia   += np.loadtxt(f"traj-{traj}/density_dia_re.dat")[:,[2,6,10]]

print(f"There were {NTraj} of {NTraj} good trajectories.")
active_state    /= NTraj
coeffs_pop_ad   /= NTraj
coeffs_pop_dia  /= NTraj


if ( NStates == 2 ):
    np.savetxt("POP_AS.dat", np.c_[ time[:,0], time[:,1], active_state[:,0], active_state[:,1] ], fmt="%1.5f")
    np.savetxt("POP_COEFF_AD.dat", np.c_[ time[:,0], time[:,1], coeffs_pop_ad[:,0], coeffs_pop_ad[:,1] ], fmt="%1.5f")
    np.savetxt("POP_COEFF_DIA.dat", np.c_[ time[:,0], time[:,1], coeffs_pop_dia[:,0], coeffs_pop_dia[:,1] ], fmt="%1.5f")
elif ( NStates == 3 ):
    np.savetxt("POP_AS.dat", np.c_[ time[:,0], time[:,1], active_state[:,0], active_state[:,1], active_state[:,2] ], fmt="%1.5f")
    np.savetxt("POP_COEFF_AD.dat", np.c_[ time[:,0], time[:,1], coeffs_pop_ad[:,0], coeffs_pop_ad[:,1], coeffs_pop_ad[:,2] ], fmt="%1.5f")
    np.savetxt("POP_COEFF_DIA.dat", np.c_[ time[:,0], time[:,1], coeffs_pop_dia[:,0], coeffs_pop_dia[:,1], coeffs_pop_dia[:,2] ], fmt="%1.5f")










