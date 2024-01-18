import numpy as np
import multiprocessing as mp


NSTATES = 2
NTRAJ   = 100#10_000
NSTEPS  = 4800
NCPUS   = NSTATES*2





def do_Parallel(jk):
    j,k = jk
    POP  = np.zeros( (NSTEPS,NSTATES) )
    TIME = 0
    for traj in range( NTRAJ ):
        TMP       = np.loadtxt("Partial__%s%s/traj-%s/population.dat" % (j,k,traj))
        TIME      = TMP[:,0]
        POP[:,:] += TMP[:,-NSTATES:]
    
    POP = POP / NTRAJ
    np.savetxt("TIME.dat", TIME)
    np.savetxt("POP_%s_%s_NTRAJ_%s.dat" % (j,k,NTRAJ), POP)


with mp.Pool(processes=NCPUS) as pool:
    pool.map(do_Parallel,[[j,k] for j in range(NSTATES) for k in range(NSTATES)])

POP = np.zeros( (NSTEPS,NSTATES) )
for j in range( NSTATES ):
    for k in range( NSTATES ):
        POP += np.loadtxt("POP_%s_%s_NTRAJ_%s.dat" % (j,k,NTRAJ))
TIME = np.loadtxt("TIME.dat")
np.savetxt("POP_TOTAL_NTRAJ_%s.dat" % (NTRAJ), np.c_[TIME, POP] )