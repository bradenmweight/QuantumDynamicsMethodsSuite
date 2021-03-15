import numpy as np
import multiprocessing as mp
import time, os
import Model_Miller_3 as model
import numpy.linalg as LA
import subprocess as sp
import random

def getGlobalParams():
    global dtE, dtI, NSteps, NTraj, NStates, M, windowtype
    global adjustedgamma, NCPUS, initstate, dirName, method
    global fs_to_au
    dtE = model.parameters.dtE
    dtI = model.parameters.dtI
    NSteps = model.parameters.NSteps
    NTraj = model.parameters.NTraj
    NStates = model.parameters.NStates
    M = model.parameters.M
    windowtype = model.parameters.windowtype.lower()
    adjustedgamma = model.parameters.adjustedgamma.lower()
    NCPUS = model.parameters.NCPUS
    initstate = model.parameters.initState
    dirName = model.parameters.dirName
    method = model.parameters.method.lower()
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
    densityFile = open(dirName+"/traj-" + str(n) + "/density.dat","w")
    if (method == "pldm"):
        coherenceFile = open(dirName+"/traj-" + str(n) + "/coherence.dat","w")
    InitCondsFile = open(dirName+"/traj-" + str(n) + "/initconds.dat","w")
    ActionMatFile = open(dirName+"/traj-" + str(n) + "/ActionMat.dat","w")
    RFile = open(dirName+"/traj-" + str(n) + "/RFile.dat","w")
    HelFile = open(dirName+"/traj-" + str(n) + "/Hel.dat","w")
    return densityFile, InitCondsFile, ActionMatFile, RFile, HelFile, coherenceFile

def closeFiles(densityFile, InitCondsFile, ActionMatFile, RFile, HelFile, coherenceFile):
    densityFile.close()    
    InitCondsFile.close()
    ActionMatFile.close()
    RFile.close()
    HelFile.close()
    coherenceFile.close()

def makeArrays():
    hist = np.zeros(( NStates ))
    rho = np.zeros(( NStates,NStates ))
    return hist,rho

def writeDensity(densityFile,i,hist,rho0=None,coherenceFile=None):

    if ( method == "pldm" ):
        rho = hist
        popList = [ round ( i * dtI ,5), round ( i * dtI / fs_to_au,5 ) ]
        coherList = [ round ( i * dtI ,5), round ( i * dtI / fs_to_au,5 ) ]
        for n in range(NStates):
            popList.append( round( (rho[n,n]*rho0).real,3 ) )
            for m in range(n+1,NStates):
                coherList.append( round( (rho[n,m]*rho0).real,3 ) )
        densityFile.write( "\t".join(map(str,popList))+"\n" )
        coherenceFile.write( "\t".join(map(str,coherList))+"\n" )

    elif ( method == "sqc" ):
        outList = [ round ( i * dtI ,5), round ( i * dtI / fs_to_au,5 ) ]
        for n in range(NStates):
            outList.append( round( hist[n],3 ) )
        densityFile.write( "\t".join(map(str,outList))+"\n" )

def writeR(R,RFile):
    RFile.write(str(round(R[0],4)) + "\n")

def writeHel(Hel,HelFile):
    outList = []
    for i in range(NStates):
        for j in range(NStates):
            outList.append( round(Hel[i,j],6) )
    HelFile.write( "\t".join(map(str,outList))+"\n" )

# Initialization of the mapping Variables
def initMapping(InitCondsFile):

    if ( method == "pldm" ):

        qp = np.zeros(( 4,NStates )) # qF, pF, qB, pB
        qF,pF,qB,pB = qp
        qF[initstate] = 1.0
        qB[initstate] = 1.0
        pF[initstate] = 1.0
        pB[initstate] = -1.0
        qp = np.array( [qF,pF,qB,pB] )

        rho0 = 0.25*(qF[initstate] - 1j*pF[initstate]) * (qB[initstate] + 1j*pB[initstate])

        outArray = []
        for n in range(NStates):
            #for m in range(NStates):
            rho = (qF[n] + 1j*pF[n]) * (qB[n] - 1j*pB[n]) * rho0
            outArray.append( (rho*rho).real )
        InitCondsFile.write( "\t".join(map(str,outArray)) + "\n")

        
        return qp, None, rho0

        
    if ( method == "sqc" ):

        ek = np.zeros((NStates))
        angle = np.zeros((NStates))
        gamma = 0 # If left at zero and no adjusted gamma, this will be delta function window.
        ZPE = np.zeros((NStates))

        if (windowtype == "square"):
            for i in range(NStates):
                ek[i] = 2*0.366*random() # [0,1]*2*gamma
                angle[i] = random() * 2 * np.pi
            
            # Shift occupied state up
            ek[initstate] += 1
                

        elif (windowtype == "n-triangle"):
            # Weighted sampling of DOF for initial state
            while (True):
                ek[initstate] = random()
                if ( 1 - ek[initstate] >= random() ):
                    break
            
            # Unoccupied DOF
            for i in range(NStates):
                angle[i] = random() * 2 * np.pi
                if (i != initstate):
                    rand = random() * ( 1 - ek[initstate] )
                    ek[i] = rand
            
            # Shift up occupied state
            ek[initstate] += 1


        ### Now we assign mapping oscillator initial conditions ###
        qF = np.zeros((NStates))
        pF = np.zeros((NStates))    

        if (adjustedgamma == "no"):
            for i in range(NStates):
                gamma = 0.36600000 * (windowtype == "square") + 0.3333333 * (windowtype == "n-triangle")
                qF[i] =  np.sqrt( 2 * ek[i] ) * np.cos(angle[i])
                pF[i] = -np.sqrt( 2 * ek[i] ) * np.sin(angle[i])
                # Action, Gamma
                InitCondsFile.write(str(ek[i] - gamma) + "\t" + str(gamma) + "\t")
            InitCondsFile.write("\n")
            return np.array( [qF,pF] ), gamma

        if (adjustedgamma == "yes"):
            for i in range(NStates):
                qF[i] =  np.sqrt( 2 * ek[i] ) * np.cos(angle[i])
                pF[i] = -np.sqrt( 2 * ek[i] ) * np.sin(angle[i])
            
            for i in range(NStates):
                ZPE[i] = ek[i] - 1 * (i == initstate)
        
            for i in range(NStates):
                # Action, ZPE
                InitCondsFile.write(str(ek[i] - ZPE[i]) + "\t" + str(ZPE[i]) + "\t")
            InitCondsFile.write("\n")
            return np.array( [qF,pF] ), ZPE

def propagateMapVars(qp, VMat):

    if ( method == "pldm" ):

        qF,pF,qB,pB = qp
        qFin, pFin, qBin, pBin = qF*1, pF*1, qB*1, pB*1
        VMatxqF =  VMat @ qF
        VMatxqB =  VMat @ qB
        pF  -= 0.5 * VMatxqF  * dtE
        pB  -= 0.5 * VMatxqB  * dtE
        qF +=  VMat @ pFin * dtE
        qB +=  VMat @ pBin * dtE
        qF -= 0.5 *  VMat @ VMatxqF * dtE**2
        qB -= 0.5 *  VMat @ VMatxqB * dtE**2
        VMatxqF =  VMat @ qF
        VMatxqB =  VMat @ qB
        pF  -= 0.5 * dtE * VMatxqF
        pB  -= 0.5 * dtE * VMatxqB

        return np.array( [qF,pF,qB,pB] ) 

    if ( method == "sqc" ):

        qF,pF = qp
        qFin, pFin = qF * 1, pF * 1
        VMatxqF =  VMat @ qF
        pF  -= 0.5 * VMatxqF  * dtE
        qF +=  VMat @ pFin * dtE
        qF -= 0.5 *  VMat @ VMatxqF * dtE**2
        VMatxqF =  VMat @ qF 
        pF  -= 0.5 * dtE * VMatxqF 
    
        return np.array( [qF,pF] ) 

def Force(dHel, R, qp, gamma_ZPE):

    if ( method == "pldm" ):

        qF,pF,qB,pB = qp
        F = np.zeros((len(R)))
        for i in range(len(qF)):
            for j in range(len(qF)):
                F -= 0.25 * dHel[i,j,:] * ( qF[i] * qF[j] + pF[i] * pF[j] + qB[i] * qB[j] + pB[i] * pB[j])
        return F

    if ( method == "sqc" ):

        qF,pF = qp
        gamma = 0
        ZPE = []
        if (adjustedgamma == "no"):
            gamma = gamma_ZPE
        elif (adjustedgamma == "yes"):
            ZPE = gamma_ZPE

        F = np.zeros((len(R)))
        if (adjustedgamma == "no"):
            for i in range(len(qF)):
                for j in range(len(qF)):
                    F -= 0.5 * dHel[i,j,:] * ( qF[i] * qF[j] + pF[i] * pF[j] - 2*gamma * (i==j) )

        elif (adjustedgamma == "yes"):
            for i in range(len(qF)):
                for j in range(len(qF)):
                    F -= 0.5 * dHel[i,j,:] * ( qF[i] * qF[j] + pF[i] * pF[j] - 2*ZPE[i] * (i==j) )
        return F

def VelVerF(R, P, qp, F1, gamma_ZPE, RFile, HelFile): # Ionic position, ionic momentum, etc.
    v = P/M
    dHel = model.dHel(R)
    F1 = Force(dHel, R, qp, gamma_ZPE)
    R += v * dtI + 0.5 * F1 * dtI ** 2 / M
    EStep = int(dtI/dtE)
    Hel = model.Hel(R)
    writeHel(Hel,HelFile)
    for t in range(EStep):
        qp = propagateMapVars(qp, Hel)
    dHel = model.dHel(R)
    F2 = Force(dHel, R, qp, gamma_ZPE)
    v += 0.5 * (F1 + F2) * dtI / M
    writeR(R,RFile)
    return R, v*M, qp, F2

 # For PLDM, not yet implemented

def getPopulation(qp, rho0):
    qF,pF,qB,pB = qp
    rho = np.zeros(( NStates,NStates ), dtype=complex) # Define density matrix
    for i in range(NStates):
       for j in range(NStates):
          rho[i,j] = (qF[i] + 1j*pF[i]) * (qB[j] - 1j*pB[j])
    return rho # Return transformed density matrix in diabatic representation

def window(qp,gamma_ZPE,ActionMatFile,densityFile,step,rho0=None,coherenceFile=None):
    hist = np.ones((NStates))

    if ( method == "pldm" ): # For PLDM, compute rho and return it

        rho = getPopulation(qp,rho0)
        writeDensity(densityFile,step,rho,rho0,coherenceFile)
        return None
    
    if ( method == "sqc" ):
    
        gamma = 0
        ZPE = []
        qF,pF = qp

        if (adjustedgamma.lower() == "no"):
            gamma = gamma_ZPE
        elif (adjustedgamma.lower() == "yes"):
            ZPE = gamma_ZPE

        if (windowtype.lower() == "square" and adjustedgamma.lower() == "no"):
            ek = np.zeros((NStates))

            for i in range(NStates):
                ek[i] = 0.5 * ( qF[i]**2 + pF[i]**2 )
            
            for i in range(NStates):
                for j in range(NStates):
                    if (i == j):
                        if ( ek[j] - 1 < 0.0 or ek[j] - 1 > 2*0.366 ):
                            hist[i] = 0
                    if (i != j):
                        if ( ek[j] < 0.0 or ek[j] > 2*0.366 ):
                            hist[i] = 0

        if (windowtype.lower() == "square" and adjustedgamma.lower() == "yes"):
            ek = np.zeros((NStates))

            for i in range(NStates):
                ek[i] = 0.5 * ( qF[i]**2 + pF[i]**2 )
            
            for i in range(NStates):
                for j in range(NStates):
                    if (i == j):
                        if ( ek[j] - 1 < 0.0 or ek[j] - 1 > 2*0.366 ):
                            hist[i] = 0
                    if (i != j):
                        if ( ek[j] < 0.0 or ek[j] > 2*0.366 ):
                            hist[i] = 0

        if (windowtype.lower() == "n-triangle" and adjustedgamma.lower() == "yes"):
            ek = np.zeros((NStates))

            for i in range(NStates):
                ek[i] = 0.5 * ( qF[i]**2 + pF[i]**2 )
            
            for i in range(NStates):
                for j in range(NStates):
                    if ( (i == j and ek[j] < 1.0) or (i != j and ek[j] >= 1.0) ):
                        hist[i] = 0
        
        if (windowtype.lower() == "n-triangle" and adjustedgamma.lower() == "no"):
            ek = np.zeros((NStates))

            for i in range(NStates):
                ek[i] = 0.5 * ( qF[i]**2 + pF[i]**2 )
            
            for i in range(NStates):
                for j in range(NStates):
                    if ( (i == j and ek[j] < 1.0) or (i != j and ek[j] >= 1.0) ):
                        hist[i] = 0

        for i in range(NStates):
            for j in range(i,NStates):
                tmp = 0
                if ( adjustedgamma.lower() == "yes" ):
                    tmp = 0.5*(qF[i]**2 + pF[j]**2) - (i==j) * ZPE[i]
                else:
                    tmp = 0.5*(qF[i]**2 + pF[j]**2) - (i==j) * gamma
                ActionMatFile.write( str( tmp ) + "\t" )
        ActionMatFile.write("\n")

        writeDensity(densityFile,step,hist)
        return None

def RunIterations(n): # This is parallelized already. "Main" for each trajectory.

    cleanDir(n)
    densityFile, InitCondsFile, ActionMatFile, RFile, HelFile, coherenceFile = initFiles(n) # Makes file objects
    hist,rho = makeArrays()

    R,P = model.initR() # Initialize nuclear DOF
    
    if (method == "sqc"):
        qp, gamma_ZPE = initMapping(InitCondsFile)
    elif (method == "pldm"):
        qp, gamma_ZPE, rho0 = initMapping(InitCondsFile)

    VMat = model.Hel(R)
    dHij = model.dHel(R)
    F1 = Force(dHij, R, qp, gamma_ZPE)
    for i in range(NSteps):
        print ("Step:", i)
        if (method == "sqc"):
            window(qp,gamma_ZPE,ActionMatFile,densityFile,i)
        elif (method == "pldm"):
            window(qp,gamma_ZPE,ActionMatFile,densityFile,i,rho0,coherenceFile)
        R, P, qp, F1 = VelVerF(R, P, qp, F1, gamma_ZPE, RFile, HelFile)
    
    closeFiles(densityFile, InitCondsFile, ActionMatFile, RFile, HelFile, coherenceFile)

    return None

def ComputeAverageDensity():
    
    # Extract printed data for each trajectory
    timeAU = np.zeros(( NSteps ))

    if ( method == "pldm" ):

        Npop = NStates
        Ncoher = int(NStates*(NStates+1)/2 - NStates)
        S = np.zeros(( Npop,NSteps )) # Population
        C = np.zeros(( Ncoher,NSteps )) # Coherence

        for t in range(NTraj):
            trajFile = open(dirName+"/traj-"+str(t)+"/density.dat","r")
            lines = np.array( [line.split() for line in trajFile.readlines()] )
            timeAU = lines[:,0].astype(float)
            for state in range( Npop ):
                S[state] += lines[:,2+state].astype(float)
            trajFile.close()

            trajFile = open(dirName+"/traj-"+str(t)+"/coherence.dat","r")
            lines = np.array( [line.split() for line in trajFile.readlines()] )
            timeAU = lines[:,0].astype(float)
            for state in range( Ncoher ):
                C[state] += lines[:,2+state].astype(float)
            trajFile.close()


        # Take average over all trajectories for each step
        S /= NTraj
        C /= NTraj

        # Normalize population at each step and print
        S_Sum = np.zeros(( NSteps ))
        averageDensityFile = open(dirName+"/density.dat","w")
        averageCoherenceFile = open(dirName+"/coherence.dat","w")
        for step in range( NSteps ):
            S_Sum[step] = np.sum(S[:,step])
            S[:,step] /= S_Sum[step]
            popList = [ round(timeAU[step],5), round(timeAU[step]/fs_to_au,5) ]
            coherList = popList*1
            for n in range(Npop):
                popList.append(round(S[n,step],3))
            for m in range(Ncoher):
                coherList.append(round(C[m,step],3))
            averageDensityFile.write( "\t".join(map(str,popList)) + "\n")
            averageCoherenceFile.write( "\t".join(map(str,coherList)) + "\n")



    if ( method == "sqc" ):

        S = np.zeros(( NStates,NSteps )) # Population

        for t in range(NTraj):
            trajFile = open(dirName+"/traj-"+str(t)+"/density.dat","r")
            lines = np.array( [line.split() for line in trajFile.readlines()] )
            timeAU = lines[:,0].astype(float)
            for state in range(NStates):
                S[state] += lines[:,2+state].astype(float)
                print (t,state,S[state,-1])
            trajFile.close()

        # Take average over all trajectories for each step
        S /= NTraj

        # Normalize each step and print
        S_Sum = np.zeros(( NSteps ))
        averageDensityFile = open(dirName+"/density.dat","w")
        for n in range( NSteps ):
            S_Sum[n] = np.sum(S[:,n])
            S[:,n] /= S_Sum[n]
            outList = [ round(timeAU[n],5), round(timeAU[n]/fs_to_au,5) ]
            for state in range(NStates):
                outList.append(round(S[state,n],3))
            averageDensityFile.write( "\t".join(map(str,outList)) + "\n")

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

    ComputeAverageDensity()
