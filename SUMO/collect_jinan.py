#!/usr/bin/env python

import os
import sys
import optparse
import subprocess
import random
import pandas as pd
import PARAMETER as p
import csv
import time
import numpy as np
import generate_route as groute
import model_v3 as cnnModel
import torch

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


def output_state_jinan_v2(step, PENETRATION_RATE, index_time):
    posMatrix = []
    velMatrix = []
    signalMatrix = []
    timeMatrix = []
    # vehicles=[]
    ori_posM = []
    ori_velM = []
    ori_time = []

    cellLength = p.CELL_LENGTH
    offset = p.OFFSET
    speedLimit = p.SPEEDLIMIT
    penetration_rate = PENETRATION_RATE
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
        timeMatrix.append([])
        ori_posM.append([])
        ori_velM.append([])
        ori_time.append([])
        signalMatrix.append(0)
        for j in range(width):
            posMatrix[i].append(0)
            ori_posM[i].append(0)
            velMatrix[i].append(0)
            ori_velM[i].append(0)
            timeMatrix[i].append(0)
            ori_time[i].append(0)
            # signalMatrix[i].append(0)

    junctionPosition = traci.junction.getPosition('0')[0]
    signals = traci.trafficlights.getRedYellowGreenState("0")
    si = 0

    for s in signals:
        if si % 4 != 3:
            index = 3 * int(si / 4) + int(si % 4)
            if s is 'G':
                signalMatrix[index] = 1
            if s is 'g':
                signalMatrix[index] = 1
        si = si + 1

    for v in vehicles_road1:
        ind = int(
            abs((junctionPosition - traci.vehicle.getPosition(v)[0] - offset)) / cellLength)
        if (ind < width):
            ori_posM[traci.vehicle.getLaneIndex(v)][width - 1 - ind] = 1
            ori_velM[traci.vehicle.getLaneIndex(v)][width - 1 - ind] = traci.vehicle.getSpeed(v) / speedLimit
            ori_time[traci.vehicle.getLaneIndex(v)][width - 1 - ind] = traci.vehicle.getWaitingTime(v) / 50

            if (random.uniform(0, 1) < penetration_rate):
                posMatrix[traci.vehicle.getLaneIndex(v)][width - 1 - ind] = 1
                velMatrix[traci.vehicle.getLaneIndex(v)][width - 1 - ind] = traci.vehicle.getSpeed(v) / speedLimit
                timeMatrix[traci.vehicle.getLaneIndex(v)][width - 1 - ind] = traci.vehicle.getWaitingTime(v) / 50
    for v in vehicles_road2:
        ind = int(
            abs((-junctionPosition + traci.vehicle.getPosition(v)[0] - offset)) / cellLength)
        if (ind < width):
            ori_posM[traci.vehicle.getLaneIndex(v) + 3][width - 1 - ind] = 1
            ori_velM[traci.vehicle.getLaneIndex(v) + 3][width - 1 - ind] = traci.vehicle.getSpeed(v) / speedLimit
            ori_time[traci.vehicle.getLaneIndex(v) + 3][width - 1 - ind] = traci.vehicle.getWaitingTime(v) / 50
            if (random.uniform(0, 1) < penetration_rate):
                posMatrix[traci.vehicle.getLaneIndex(v) + 3][width - 1 - ind] = 1
                velMatrix[traci.vehicle.getLaneIndex(v) + 3][width - 1 - ind] = traci.vehicle.getSpeed(v) / speedLimit
                timeMatrix[traci.vehicle.getLaneIndex(v) + 3][width - 1 - ind] = traci.vehicle.getWaitingTime(v) / 50

    junctionPosition = traci.junction.getPosition('0')[1]

    for v in vehicles_road3:
        ind = int(
            abs((junctionPosition - traci.vehicle.getPosition(v)[1] - offset)) / cellLength)
        if (ind < width):
            ori_posM[traci.vehicle.getLaneIndex(v) + 6][width - 1 - ind] = 1
            ori_velM[traci.vehicle.getLaneIndex(v) + 6][width - 1 - ind] = traci.vehicle.getSpeed(v) / speedLimit
            ori_time[traci.vehicle.getLaneIndex(v) + 6][width - 1 - ind] = traci.vehicle.getWaitingTime(v) / 50
            if (random.uniform(0, 1) < penetration_rate):
                posMatrix[traci.vehicle.getLaneIndex(v) + 6][width - 1 - ind] = 1
                velMatrix[traci.vehicle.getLaneIndex(v) + 6][width - 1 - ind] = traci.vehicle.getSpeed(v) / speedLimit
                timeMatrix[traci.vehicle.getLaneIndex(v) + 6][width - 1 - ind] = traci.vehicle.getWaitingTime(v) / 50
    for v in vehicles_road4:
        ind = int(
            abs((-junctionPosition + traci.vehicle.getPosition(v)[1] - offset)) / cellLength)
        if (ind < width):
            ori_posM[traci.vehicle.getLaneIndex(v) + 9][width - 1 - ind] = 1
            ori_velM[traci.vehicle.getLaneIndex(v) + 9][width - 1 - ind] = traci.vehicle.getSpeed(v) / speedLimit
            ori_time[traci.vehicle.getLaneIndex(v) + 9][width - 1 - ind] = traci.vehicle.getWaitingTime(v) / 50
            if (random.uniform(0, 1) < penetration_rate):
                posMatrix[traci.vehicle.getLaneIndex(v) + 9][width - 1 - ind] = 1
                velMatrix[traci.vehicle.getLaneIndex(v) + 9][width - 1 - ind] = traci.vehicle.getSpeed(v) / speedLimit
                timeMatrix[traci.vehicle.getLaneIndex(v) + 9][width - 1 - ind] = traci.vehicle.getWaitingTime(v) / 50

    arrivalrate = '_jinan' + str(index_time)

    stime = time.strftime("%m-%d-%H-%M-%S", time.localtime())
    filename_1 = 'data/output2/vel_' + str(int(PENETRATION_RATE * 100)) + arrivalrate + '_' + stime + '_' + str(
        step) + '.csv'
    vall_f = pd.DataFrame(velMatrix)
    vall_f.to_csv(filename_1, mode='w')

    filename_2 = 'data/output2/pos_' + str(int(PENETRATION_RATE * 100)) + arrivalrate + '_' + stime + '_' + str(
        step) + '.csv'
    vall_f = pd.DataFrame(posMatrix)
    vall_f.to_csv(filename_2, mode='w')

    filename_3 = 'data/output2/tim_' + str(int(PENETRATION_RATE * 100)) + arrivalrate + '_' + stime + '_' + str(
        step) + '.csv'
    vall_f = pd.DataFrame(timeMatrix)
    vall_f.to_csv(filename_3, mode='w')

    filename_4 = 'data/output2/vel' + "S" + str(int(PENETRATION_RATE * 100)) + arrivalrate + '_' + stime + '_' + str(
        step) + '.csv'
    # svall_f = pd.DataFrame(ori_velM)
    # svall_f.to_csv(filename_4, mode='w')

    filename_5 = 'data/output2/pos' + "S" + str(int(PENETRATION_RATE * 100)) + arrivalrate + '_' + stime + '_' + str(
        step) + '.csv'
    svall_f = pd.DataFrame(ori_posM)
    svall_f.to_csv(filename_5, mode='w')

    filename_6 = 'data/output2/tim' + "S" + str(int(PENETRATION_RATE * 100)) + arrivalrate + '_' + stime + '_' + str(
        step) + '.csv'
    svall_f = pd.DataFrame(ori_time)
    svall_f.to_csv(filename_6, mode='w')

    filename_7 = 'data/output2/signal' + str(int(PENETRATION_RATE * 100)) + arrivalrate + '_' + stime + '_' + str(
        step) + '.csv'
    svall_f = pd.DataFrame(signalMatrix)
    svall_f.to_csv(filename_7, mode='w')

    filename_8 = 'data/output2/sumo_trainingset.csv'

    with open(filename_8, "a+") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [filename_1, filename_2, filename_3, filename_4, filename_5, filename_6, filename_7, PENETRATION_RATE])


def output_weight_jinan_v3(step, PENETRATION_RATE):
    t1 = time.clock()
    posMatrix = []
    velMatrix = []
    signalMatrix = []
    timeMatrix = []
    # vehicles=[]
    # ori_posM = []
    # ori_velM = []
    # ori_time = []

    cellLength = p.CELL_LENGTH
    offset = p.OFFSET
    speedLimit = p.SPEEDLIMIT
    penetration_rate = PENETRATION_RATE
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
        timeMatrix.append([])
        # ori_posM.append([])
        # ori_velM.append([])
        # ori_time.append([])
        signalMatrix.append(0)
        for j in range(width):
            posMatrix[i].append(0)
            # ori_posM[i].append(0)
            velMatrix[i].append(0)
            # ori_velM[i].append(0)
            timeMatrix[i].append(0)
            # ori_time[i].append(0)
            # signalMatrix[i].append(0)

    junctionPosition = traci.junction.getPosition('0')[0]
    signals = traci.trafficlights.getRedYellowGreenState("0")
    si = 0

    for s in signals:
        if si % 4 != 3:
            index = 3 * int(si / 4) + int(si % 4)
            if s is 'G':
                signalMatrix[index] = 1
            if s is 'g':
                signalMatrix[index] = 1
        si = si + 1

    for v in vehicles_road1:
        ind = int(
            abs((junctionPosition - traci.vehicle.getPosition(v)[0] - offset)) / cellLength)
        # if (ind < width):
        #     ori_posM[traci.vehicle.getLaneIndex(v)][width - 1 - ind] = 1
        #     ori_velM[traci.vehicle.getLaneIndex(v)][width - 1 - ind] = traci.vehicle.getSpeed(v) / speedLimit
        #     ori_time[traci.vehicle.getLaneIndex(v)][width - 1 - ind] = traci.vehicle.getWaitingTime(v) / 50

        if traci.vehicle.getTypeID(v) == "CAV_TYPE":
            # if (random.uniform(0, 1) < penetration_rate):
            posMatrix[traci.vehicle.getLaneIndex(v)][width - 1 - ind] = 1
            velMatrix[traci.vehicle.getLaneIndex(v)][width - 1 - ind] = traci.vehicle.getSpeed(v) / speedLimit
            timeMatrix[traci.vehicle.getLaneIndex(v)][width - 1 - ind] = traci.vehicle.getWaitingTime(v) / 50
    for v in vehicles_road2:
        ind = int(
            abs((-junctionPosition + traci.vehicle.getPosition(v)[0] - offset)) / cellLength)
        # if (ind < width):
        #     ori_posM[traci.vehicle.getLaneIndex(v) + 3][width - 1 - ind] = 1
        #     ori_velM[traci.vehicle.getLaneIndex(v) + 3][width - 1 - ind] = traci.vehicle.getSpeed(v) / speedLimit
        #     ori_time[traci.vehicle.getLaneIndex(v) + 3][width - 1 - ind] = traci.vehicle.getWaitingTime(v) / 50
        if (random.uniform(0, 1) < penetration_rate):
            posMatrix[traci.vehicle.getLaneIndex(v) + 3][width - 1 - ind] = 1
            velMatrix[traci.vehicle.getLaneIndex(v) + 3][width - 1 - ind] = traci.vehicle.getSpeed(v) / speedLimit
            timeMatrix[traci.vehicle.getLaneIndex(v) + 3][width - 1 - ind] = traci.vehicle.getWaitingTime(v) / 50

    junctionPosition = traci.junction.getPosition('0')[1]

    for v in vehicles_road3:
        ind = int(
            abs((junctionPosition - traci.vehicle.getPosition(v)[1] - offset)) / cellLength)
        # if (ind < width):
        #     ori_posM[traci.vehicle.getLaneIndex(v) + 6][width - 1 - ind] = 1
        #     ori_velM[traci.vehicle.getLaneIndex(v) + 6][width - 1 - ind] = traci.vehicle.getSpeed(v) / speedLimit
        #     ori_time[traci.vehicle.getLaneIndex(v) + 6][width - 1 - ind] = traci.vehicle.getWaitingTime(v) / 50
        if (random.uniform(0, 1) < penetration_rate):
            posMatrix[traci.vehicle.getLaneIndex(v) + 6][width - 1 - ind] = 1
            velMatrix[traci.vehicle.getLaneIndex(v) + 6][width - 1 - ind] = traci.vehicle.getSpeed(v) / speedLimit
            timeMatrix[traci.vehicle.getLaneIndex(v) + 6][width - 1 - ind] = traci.vehicle.getWaitingTime(v) / 50
    for v in vehicles_road4:
        ind = int(
            abs((-junctionPosition + traci.vehicle.getPosition(v)[1] - offset)) / cellLength)
        # if (ind < width):
        #     ori_posM[traci.vehicle.getLaneIndex(v) + 9][width - 1 - ind] = 1
        #     ori_velM[traci.vehicle.getLaneIndex(v) + 9][width - 1 - ind] = traci.vehicle.getSpeed(v) / speedLimit
        #     ori_time[traci.vehicle.getLaneIndex(v) + 9][width - 1 - ind] = traci.vehicle.getWaitingTime(v) / 50
        if (random.uniform(0, 1) < penetration_rate):
            posMatrix[traci.vehicle.getLaneIndex(v) + 9][width - 1 - ind] = 1
            velMatrix[traci.vehicle.getLaneIndex(v) + 9][width - 1 - ind] = traci.vehicle.getSpeed(v) / speedLimit
            timeMatrix[traci.vehicle.getLaneIndex(v) + 9][width - 1 - ind] = traci.vehicle.getWaitingTime(v) / 50
    tposMatrix = torch.tensor(posMatrix, dtype=torch.float64)
    tvelMatrix = torch.tensor(velMatrix, dtype=torch.float64)
    ttimeMatrix = torch.tensor(timeMatrix, dtype=torch.float64)
    tsignalMatrix = torch.tensor(signalMatrix, dtype=torch.float64)

    input1 = torch.stack([tvelMatrix, tposMatrix, ttimeMatrix])
    # print input1.size()
    input1 = input1.unsqueeze(0)
    input2 = tsignalMatrix
    input2 = input2.unsqueeze(0)

    model = cnnModel.CNNv4()
    model = model.double()
    model.load_state_dict(torch.load('data/model/model42_04293.pkl'))
    model.eval()
    output = model(input1, input2)

    # output = torch.reshape(output, [1, 12, 50])

    output = torch.reshape(output, [1, 1, 12])
    output_2 = output.type(torch.torch.ShortTensor)

    queuelength = []
    for i in range(9):
        queuelength.append(0)
    queuelength.append(0)
    # waiting = []

    tra_ft2w = {1: 11, 2: 10, 3: 2, 4: 1, 5: 8, 6: 7, 7: 5, 8: 4}
    for i in range(8):
        # if i % 3 != 0:
        queuelength[i + 1] = output_2[0][0][(tra_ft2w[i + 1])].item()

    # queuelength = output.ge(0.01)
    # veh_num = []
    # wat_num = []
    # veh_num.append(0)
    # wat_num.append(0)
    # for i in range(12):
    #     if i%3!=0:
    #         veh_num.append(torch.sum(queuelength[0][i][:]).item())
    #         wat_num.append(torch.sum(output[0][i][:]).item())
    #     if i%3==2:
    #         tmp2=veh_num.pop()
    #         tmp1=veh_num.pop()
    #         veh_num.append(tmp2)
    #         veh_num.append(tmp1)
    #
    #         tmp2=wat_num.pop()
    #         tmp1=wat_num.pop()
    #         wat_num.append(tmp2)
    #         wat_num.append(tmp1)
    #
    #
    t2 = time.clock()
    #
    # tmp=wat_num[7:8]
    # wat_num[3:4]=tmp
    #
    # tmp=veh_num[3]
    # veh_num[3]=veh_num[7]
    # veh_num[7]=tmp
    #
    # tmp = veh_num[4]
    # veh_num[4] = veh_num[8]
    # veh_num[8]=tmp

    print 'Processing time of CNN:', t2 - t1

    return queuelength


def run_collectdata_jinan(penetration_rate, index_time):
    traci.init(PORT)
    step = 0
    # we start with phase 2 where EW has green
    traci.trafficlights.setPhase("0", 0)
    steps = np.random.randint(3200, size=200)
    steps = steps.tolist()

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        # if step%15==0:
        # output_weight_v3(step, penetration_rate)
        if step in steps:
            output_state_jinan_v2(step, penetration_rate, index_time)

        # phasesrecord.append(traci.trafficlights.getPhase("0"))

        # run.updateWeight('w')

        step += 1
    traci.close()
    sys.stdout.flush()
    return step


def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = optParser.parse_args()
    return options


def run_main(p, index):
    options = get_options()
    # this script has been called from the command line. It will start sumo as a
    # server, then connect and run
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    # first, generate the route file for this simulation
    groute.generate_routefile_v2("va_data/cross.rou.xml", index)

    # groute.generate_routefile_v1("va_data/cross.rou.xml")
    # this is the normal way of using traci. sumo is started as a
    # subprocess and then the python script connects and runs
    sumoProcess = subprocess.Popen([sumoBinary, "-c", "va_data/cross.sumocfg", "--summary-output",
                                    "OUTPUT/tripinfo_va.xml", "--remote-port", str(PORT)], stdout=sys.stdout,
                                   stderr=sys.stderr)
    step = run_collectdata_jinan(p, index)
    # sumoProcess.kill()


if __name__ == "__main__":

    for i in range(64):
        run_main(abs(0.5 + 0.1 * np.random.randn()), i)

    # run_main(0.7)
