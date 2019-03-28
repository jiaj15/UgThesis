#!/usr/bin/env python

import os
import sys
import optparse
import subprocess
import random
import pandas as pd
import PARAMETER as p

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


def generate_routefile():
    random.seed(42)  # make tests reproducible
    N = 3600  # number of time steps
    # demand per second from different directions

    with open("data/cross.rou.xml", "w") as routes:
        print >> routes, """<routes>
        <vType id="SUMO_DEFAULT_TYPE" accel="0.8" decel="4.5" sigma="0" length="5" minGap="2" maxSpeed="70"/>
       
        <route id="W2E" edges="51o 1i 2o 52i" />
        <route id="E2W" edges="52o 2i 1o 51i" />
        <route id="N2S" edges="54o 4i 3o 53i" />
        <route id="S2N" edges="53o 3i 4o 54i" />
        
        <route id="W2N" edges="51o 1i 4o 54i" />
        <route id="N2E" edges="54o 4i 2o 52i" />
        <route id="E2S" edges="52o 2i 3o 53i" />
        <route id="S2W" edges="53o 3i 1o 51i" />
        
        <route id="W2S" edges="51o 1i 3o 53i" />
        <route id="S2E" edges="53o 3i 2o 52i" />
        <route id="E2N" edges="52o 2i 4o 54i" />
        <route id="N2W" edges="54o 4i 1o 51i" />
        
        """
        lastVeh = 0
        vehNr = 0
        for i in range(N):
            # ZHIXING
            if random.uniform(0, 1) < p.pWE:
                print >> routes, '    <vehicle id="right_%i" type="SUMO_DEFAULT_TYPE" route="W2E" depart="%i" />' % (
                    vehNr, i)
                vehNr += 1
                lastVeh = i
            if random.uniform(0, 1) < p.pEW:
                print >> routes, '    <vehicle id="left_%i" type="SUMO_DEFAULT_TYPE" route="E2W" depart="%i" />' % (
                    vehNr, i)
                vehNr += 1
                lastVeh = i
            if random.uniform(0, 1) < p.pNS:
                print >> routes, '    <vehicle id="down_%i" type="SUMO_DEFAULT_TYPE" route="N2S" depart="%i" color="1,0,0"/>' % (
                    vehNr, i)
                vehNr += 1
                lastVeh = i
            if random.uniform(0, 1) < p.pSN:
                print >> routes, '    <vehicle id="up_%i" type="SUMO_DEFAULT_TYPE" route="S2N" depart="%i" color="1,0,0"/>' % (
                    vehNr, i)
                vehNr += 1
                lastVeh = i

            # ZUO ZHUAN
            if random.uniform(0, 1) < p.pWN:
                print >> routes, '    <vehicle id="rightLT_%i" type="SUMO_DEFAULT_TYPE" route="W2E" depart="%i" />' % (
                    vehNr, i)
                vehNr += 1
                lastVeh = i
            if random.uniform(0, 1) < p.pNE:
                print >> routes, '    <vehicle id="leftLT_%i" type="SUMO_DEFAULT_TYPE" route="E2W" depart="%i" />' % (
                    vehNr, i)
                vehNr += 1
                lastVeh = i
            if random.uniform(0, 1) < p.pES:
                print >> routes, '    <vehicle id="downLT_%i" type="SUMO_DEFAULT_TYPE" route="N2S" depart="%i" color="1,0,0"/>' % (
                    vehNr, i)
                vehNr += 1
                lastVeh = i
            if random.uniform(0, 1) < p.pSW:
                print >> routes, '    <vehicle id="upLT_%i" type="SUMO_DEFAULT_TYPE" route="S2N" depart="%i" color="1,0,0"/>' % (
                    vehNr, i)
                vehNr += 1
                lastVeh = i

            # YOU ZHUAN
            if random.uniform(0, 1) < p.pWS:
                print >> routes, '    <vehicle id="rightRL_%i" type="SUMO_DEFAULT_TYPE" route="W2E" depart="%i" />' % (
                    vehNr, i)
                vehNr += 1
                lastVeh = i
            if random.uniform(0, 1) < p.pSE:
                print >> routes, '    <vehicle id="leftRL_%i" type="SUMO_DEFAULT_TYPE" route="E2W" depart="%i" />' % (
                    vehNr, i)
                vehNr += 1
                lastVeh = i
            if random.uniform(0, 1) < p.pEN:
                print >> routes, '    <vehicle id="downRL_%i" type="SUMO_DEFAULT_TYPE" route="N2S" depart="%i" color="1,0,0"/>' % (
                    vehNr, i)
                vehNr += 1
                lastVeh = i
            if random.uniform(0, 1) < p.pNW:
                print >> routes, '    <vehicle id="upRL_%i" type="SUMO_DEFAULT_TYPE" route="S2N" depart="%i" color="1,0,0"/>' % (
                    vehNr, i)
                vehNr += 1
                lastVeh = i

        print >> routes, "</routes>"


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


def run():
    """execute the TraCI control loop"""
    traci.init(PORT)
    step = 0
    # we start with phase 2 where EW has green
    traci.trafficlights.setPhase("0", 2)
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        output_state(step)
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
    generate_routefile()

    # this is the normal way of using traci. sumo is started as a
    # subprocess and then the python script connects and runs
    sumoProcess = subprocess.Popen([sumoBinary, "-c", "data/cross.sumocfg", "--tripinfo-output",
                                    "tripinfo.xml", "--remote-port", str(PORT)], stdout=sys.stdout, stderr=sys.stderr)
    run()
    sumoProcess.wait()
