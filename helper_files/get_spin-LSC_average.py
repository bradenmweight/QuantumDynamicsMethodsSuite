import numpy as np
from glob import glob

def getPartial( ):
    traj_list = glob(f"traj-*")

    data = np.loadtxt( f"{traj_list[0]}/density.dat" ) * 0.0
    counter = 0
    
    #for traj in range( NTraj ):
    for traj in range( len(traj_list) ):
        #print (f"Working on traj = {traj}")
        try:
            tmp = np.loadtxt( f"{traj_list[traj]}/density.dat" )
            if ( len(tmp) > 0 ):
                data += tmp
            else:
                continue
        except:
            continue
        counter += 1
    data /= counter
    print (f"# Bad = {len(traj_list) - counter}  # Good = {counter}")
    np.savetxt( f"density.dat", data )

def main():
    getPartial()

if ( __name__ == "__main__"):
    main()