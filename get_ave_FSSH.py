import numpy as np
#from matplotlib import pyplot as plt
import math

NTraj = 10 ** 3
NStates = 3

time           = np.loadtxt("traj-0/active_state.dat")[:,[0,1]]
NSteps         = len( time )
coeffs_pop_ad  = np.zeros(( NSteps, NStates ))
coeffs_pop_dia = np.zeros(( NSteps, NStates ))
active_state   = np.zeros(( NSteps, NStates ))


def active_state_populations( integers, active_state ):
    for step in range( NSteps ):
        index = integers[step]
        #print( step, index, active_state[step, :] )
        active_state[step,index]  += 1
    return active_state

for traj in range( NTraj ):
    print( f"traj {traj} of {NTraj}" )
    #integers          = np.loadtxt(f"traj-{traj}/active_state.dat")[:,2].astype(int)
    #active_state      = active_state_populations( integers, active_state )
    active_state     += np.loadtxt(f"traj-{traj}/active_state.dat")[:,2:]
    coeffs_pop_ad    += np.loadtxt(f"traj-{traj}/density_dia_re.dat")[:,[2,6,10]]
    coeffs_pop_dia   += np.loadtxt(f"traj-{traj}/density_ad_re.dat")[:,[2,6,10]]


#for step in range( NSteps ):
    #active_state[step,:] /= np.sum( active_state[step,:] )
active_state    /= NTraj
coeffs_pop_ad   /= NTraj
coeffs_pop_dia  /= NTraj

np.savetxt("POP_AS.dat", np.c_[ time[:,0], time[:,1], active_state[:,0], active_state[:,1], active_state[:,2] ], fmt="%1.5f")
np.savetxt("POP_COEFF_AD.dat", np.c_[ time[:,0], time[:,1], coeffs_pop_ad[:,0], coeffs_pop_ad[:,1], coeffs_pop_ad[:,2] ], fmt="%1.5f")
np.savetxt("POP_COEFF_DIA.dat", np.c_[ time[:,0], time[:,1], coeffs_pop_dia[:,0], coeffs_pop_dia[:,1], coeffs_pop_dia[:,2] ], fmt="%1.5f")

