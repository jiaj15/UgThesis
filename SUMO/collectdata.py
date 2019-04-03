#!/usr/bin/env python

import os
import sys
import optparse
import subprocess
import random
import pandas as pd
import PARAMETER as p
import graph
random.seed(42)  # make tests reproducible
# we need to import python modules from the $SUMO_HOME/tools directory
try:
    sys.path.append(os.path.join(os.path.dirname(
        __file__), '..', '..', '..', '..', "tools"))  # tutorial in tests
    sys.path.append(os.path.join(os.environ.get("SUMO_HOME", os.path.join(
        os.path.dirname(__file__), "..", "..", "..")), "tools"))  # tutorial in docs
    from sumolib import checkBinary
except ImportError:
    sys.exit(
        "please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')")

import traci

# the port used for communicating with your sumo instance
PORT = 8873





# The program looks like this
#    <tlLogic id="0" type="static" programID="0" offset="0">
# the locations of the tls are      NESW
#        <phase duration="31" state="GrGr"/>
#        <phase duration="6"  state="yryr"/>
#        <phase duration="31" state="rGrG"/>
#        <phase duration="6"  state="ryry"/>
#    </tlLogic>

def output_state(step):
    posMatrix = []
    velMatrix = []
    # vehicles=[]
    ori_posM = []
    ori_velM = []

    cellLength = p.CELL_LENGTH
    offset = p.OFFSET
    speedLimit = p.SPEEDLIMIT
    penetration_rate = p.PENETRATION_RATE
    width = p.WIDTH

    currenttime = step  # traci.SimulationDomain.getCurrentTime()

    vehicles_road1 = traci.edge.getLastStepVehicleIDs('1i')
    vehicles_road2 = traci.edge.getLastStepVehicleIDs('2i')
    vehicles_road3 = traci.edge.getLastStepVehicleIDs('3i')
    vehicles_road4 = traci.edge.getLastStepVehicleIDs('4i')

    # for index in range(4):
    #     edge_index=str(index+1)+'i'
    #     vehicles.append(traci.edge.getLastStepVehicleIDs(edge_index))
    for i in range(12):
        posMatrix.append([])
        velMatrix.append([])
        ori_posM.append([])
        ori_velM.append([])
        for j in range(width):
            posMatrix[i].append(0)
            ori_posM[i].append(0)
            velMatrix[i].append(0)
            ori_velM[i].append(0)

    junctionPosition = traci.junction.getPosition('0')[0]

    for v in vehicles_road1:
        ind = int(
            abs((junctionPosition - traci.vehicle.getPosition(v)[0] - offset)) / cellLength)
        if (ind < width):
            ori_posM[traci.vehicle.getLaneIndex(v)][width - 1 - ind] = 1
            ori_velM[traci.vehicle.getLaneIndex(v)][width - 1 - ind] = traci.vehicle.getSpeed(v) / speedLimit

            if (random.uniform(0, 1) < penetration_rate):
                posMatrix[traci.vehicle.getLaneIndex(v)][width - 1 - ind] = 1
                velMatrix[traci.vehicle.getLaneIndex(v)][width - 1 - ind] = traci.vehicle.getSpeed(v) / speedLimit
    for v in vehicles_road2:
        ind = int(
            abs((-junctionPosition + traci.vehicle.getPosition(v)[0] - offset)) / cellLength)
        if (ind < width):
            ori_posM[traci.vehicle.getLaneIndex(v) + 3][width - 1 - ind] = 1
            ori_velM[traci.vehicle.getLaneIndex(v) + 3][width - 1 - ind] = traci.vehicle.getSpeed(v) / speedLimit
            if (random.uniform(0, 1) < penetration_rate):
                posMatrix[traci.vehicle.getLaneIndex(v) + 3][width - 1 - ind] = 1
                velMatrix[traci.vehicle.getLaneIndex(v) + 3][width - 1 - ind] = traci.vehicle.getSpeed(v) / speedLimit

    junctionPosition = traci.junction.getPosition('0')[1]

    for v in vehicles_road3:
        ind = int(
            abs((junctionPosition - traci.vehicle.getPosition(v)[1] - offset)) / cellLength)
        if (ind < width):
            ori_posM[traci.vehicle.getLaneIndex(v) + 6][width - 1 - ind] = 1
            ori_velM[traci.vehicle.getLaneIndex(v) + 6][width - 1 - ind] = traci.vehicle.getSpeed(v) / speedLimit
            if (random.uniform(0, 1) < penetration_rate):
                posMatrix[traci.vehicle.getLaneIndex(v) + 6][width - 1 - ind] = 1
                velMatrix[traci.vehicle.getLaneIndex(v) + 6][width - 1 - ind] = traci.vehicle.getSpeed(v) / speedLimit
    for v in vehicles_road4:
        ind = int(
            abs((-junctionPosition + traci.vehicle.getPosition(v)[1] - offset)) / cellLength)
        if (ind < width):
            ori_posM[traci.vehicle.getLaneIndex(v) + 9][width - 1 - ind] = 1
            ori_velM[traci.vehicle.getLaneIndex(v) + 9][width - 1 - ind] = traci.vehicle.getSpeed(v) / speedLimit
            if (random.uniform(0, 1) < penetration_rate):
                posMatrix[traci.vehicle.getLaneIndex(v) + 9][width - 1 - ind] = 1
                velMatrix[traci.vehicle.getLaneIndex(v) + 9][width - 1 - ind] = traci.vehicle.getSpeed(v) / speedLimit

    # dirname='data/output/'+str(currenttime)
    # os.makedirs(dirname)

    # filename=dirname+'/pos.csv'
    # posm_f=pd.DataFrame(posMatrix)
    # posm_f.to_csv(filename)
    #
    # filename=dirname+'/vel.csv'
    # velm_f=pd.DataFrame(velMatrix)
    # velm_f.to_csv(filename)
    #
    # filename=dirname+'/tls.csv'
    # tls_f=pd.DataFrame(traci.trafficlights.getPhase("0"))
    # tls_f.to_csv(filename)

    filename = 'data/output/pos_pr.csv'
    posm_f = pd.DataFrame({step: posMatrix})
    posm_f.to_csv(filename, mode="a+")

    filename = 'data/output/pos_or.csv'
    posm_f = pd.DataFrame({step: ori_posM})
    posm_f.to_csv(filename, mode="a+")

    filename = 'data/output/vel_pr.csv'
    posm_f = pd.DataFrame({step: velMatrix})
    posm_f.to_csv(filename, mode="a+")

    filename = 'data/output/pos_or.csv'
    posm_f = pd.DataFrame({step: ori_velM})
    posm_f.to_csv(filename, mode="a+")

def updateweightGetTraci():
    #weight=[0, 0, 0, 0, 0, 0, 0, 0, 0]
    volume=[]
    volume.append(0)

    volume.append(traci.lane.getLastStepVehicleNumber("4i_2"))
    volume.append(traci.lane.getLastStepVehicleNumber("4i_1"))

    volume.append(traci.lane.getLastStepVehicleNumber("1i_2"))
    volume.append(traci.lane.getLastStepVehicleNumber("1i_1"))

    volume.append(traci.lane.getLastStepVehicleNumber("3i_2"))
    volume.append(traci.lane.getLastStepVehicleNumber("3i_1"))

    volume.append(traci.lane.getLastStepVehicleNumber("2i_2"))
    volume.append(traci.lane.getLastStepVehicleNumber("2i_1"))

    #


    print volume

    orlight = 'grrrgrrrgrrrgrrr'
    orlight = list(orlight)

    trans = {1:[2,3],2:[1],3:[14,15],4:[13],5:[10,11],6:[9],7:[6,7],8:[5]}

    minst=graph.minst(volume)

    print minst

    for i in minst:
        for j in trans[i]:
            orlight[j]='G'
    light=orlight[0]
    for k in range(1,len(orlight)):
        light =light + orlight[k]




    print light
    return light

def transTraci(light,prelight):
    light=list(light)
    prelight=list(prelight)
    translight=''
    for i in range(len(light)):
        if light[i]== prelight[i]:
            translight=translight+light[i]
        else:
            if prelight[i]=='G':
                translight=translight+'y'
            else:
                translight=translight+light[i]

    return translight








def run():
    """execute the TraCI control loop"""
    traci.init(PORT)
    step = 0
    # we start with phase 2 where EW has green
    traci.trafficlights.setPhase("0", 0)
    curlight='grrrgrrrgrrrgrrr'
    while traci.simulation.getMinExpectedNumber() > 0:
        if step%25==0:
            nextlight=updateweightGetTraci()
            #curlight=transTraci(nextlight,curlight) #yellow state
        else:
            curlight=nextlight
        traci.trafficlights.setRedYellowGreenState("0",curlight)
        #traci.trafficlights.setPhase("0", 0)
        #updateweight()
        traci.simulationStep()
        # output_state(step)
        # if traci.trafficlights.getPhase("0") == 2:
        #     if step % 500 > 250:
        #         traci.trafficlights.setPhase("0", 3)
        #     else:
        #         traci.trafficlights.setPhase("0", 2)
        # we are not already switching
        # if traci.inductionloop.getLastStepVehicleNumber("0") > 0:
        #     # there is a vehicle from the north, switch
        #     traci.trafficlights.setPhase("0", 3)
        # else:
        #     # otherwise try to keep green for EW
        #     traci.trafficlights.setPhase("0", 2)
        step += 1
    traci.close()
    sys.stdout.flush()


def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = optParser.parse_args()
    return options


# this is the main entry point of this script
if __name__ == "__main__":
    options = get_options()

    # this script has been called from the command line. It will start sumo as a
    # server, then connect and run
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    # first, generate the route file for this simulation
    #generate_routefile()

    # this is the normal way of using traci. sumo is started as a
    # subprocess and then the python script connects and runs
    sumoProcess = subprocess.Popen([sumoBinary, "-c", "data/cross.sumocfg", "--tripinfo-output",
                                    "tripinfo.xml", "--remote-port", str(PORT)], stdout=sys.stdout, stderr=sys.stderr)
    run()
    sumoProcess.wait()
