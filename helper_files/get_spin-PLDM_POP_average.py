import numpy as np
import multiprocessing as mp
from glob import glob

def getGlobals():
    global N, NTraj, NSkip, NCPUS
    N = 7
    NTraj = 10 ** 5
    NSkip = 1 # Probably already skipped some in calculation
    NCPUS = 49

def getCombinations():
    pairs = []
    for j in range( N ):
        for k in range( N ):
            pairs.append( [j,k] )
    return pairs

def getPartial( pair ):
    j,k = pair
    #if ( j != 1 and k != 1 ): return
    print (f"Working on {j}{k} part.")

    traj_list = glob(f"Partial__{j}{k}/traj-*")

    data = np.loadtxt( f"{traj_list[0]}/population.dat" ) * 0.0
    counter = 0
    
    #for traj in range( NTraj ):
    for traj in range( len(traj_list) ):
        #print (f"Working on traj = {traj}")
        try:
            data += np.loadtxt( f"{traj_list[traj]}/population.dat" )
        except:
            continue
        counter += 1
    data /= counter
    print (f"Partial {j}{k}: # Bad = {NTraj - counter}  # Good = {counter}")
    np.savetxt( f"population_Partial_{j}{k}.dat", data )

def getTotal():

    data = np.loadtxt( f"population_Partial_11.dat" ) * 0.0
    counter = 0
    for j in range( N ):
        for k in range( N ):
            try:
                data += np.loadtxt( f"population_Partial_{j}{k}.dat" )
    #data += 2 * np.loadtxt( f"population_Partial_01.dat" )
            except:
                continue
            counter += 1
    data[:,:2] /= N**2 # Restore time
    np.savetxt( f"population_Total.dat", data )





def main():

    getGlobals()
    pairs = getCombinations()
    with mp.Pool( processes=NCPUS ) as pool:
        pool.map(getPartial,pairs)
    getTotal()

if ( __name__ == "__main__"):
    main()