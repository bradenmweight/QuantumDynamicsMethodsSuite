# The purpose of this script is to get the population
#   density from multiple trajectories and average them
###             ~ Braden Weight ~
###             June 13, 2022

import numpy as np

NTraj = 10 ** 4 # Number of trajectories
NSteps = 1034 # Number of steps per trajectory
NStates = 2 # Dimension of reduced density matrix

# Get all histogram/density data from trajectory folders
density = np.zeros((NTraj,NSteps,NStates))
time = np.zeros((NSteps))
for t in range(NTraj):
    print (t, "of", NTraj)
    file01 = open("traj-" + str(t) + "/density.dat","r")
    for count, line in enumerate(file01):
        s = line.split()
        if (count < NSteps and len(s) == NStates+1):
            time[count] = float(s[0])
            for state in range( NStates ):
                density[t,count,state] += float(s[state+1])
    file01.close()

# Get the average over all trajectories at each step
average_density = np.zeros((NSteps,NStates))
for n in range(NSteps):
    for state in range( NStates ):
        sum1 = np.sum(density[:,n,state]) 
        average_density[n,state] = sum1 / NTraj
    sumP = np.sum( average_density[n,:] )
    for state in range( NStates ):
        average_density[n,state] /= sumP

# Write results to file
file01 = open(f"density_{NTraj}.dat","w")
for n in range(NSteps):
    #line = np.round( np.array([time[n],time[n]/41.341,average_density[n,0],average_density[n,1],average_density[n,2]]).astype(float) ,4)
    line = [time[n],time[n]/41.341]
    for state in range( NStates ):
        line.append( average_density[n,state] )    
    np.round( np.array(line) ,4)
    file01.write( "\t".join(map(str,line)) + "\n" )
file01.close()

