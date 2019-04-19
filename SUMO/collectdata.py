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
import model_v2 as cnnModel
import torch

# random.seed(42)  # make tests reproducible
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


def output_state_v1(step, PENETRATION_RATE):
    posMatrix = []
    velMatrix = []
    signalMatrix = []
    #     timeMatrix=[]
    # vehicles=[]
    ori_posM = []
    ori_velM = []
    #     ori_time=[]

    cellLength = p.CELL_LENGTH
    offset = p.OFFSET
    speedLimit = p.SPEEDLIMIT
    penetration_rate = PENETRATION_RATE
    width = p.WIDTH

    currenttime = step  # traci.SimulationDomain.getCurrentTime()

    vehicles_road1 = traci.edge.getLastStepVehicleIDs('4i')
    vehicles_road2 = traci.edge.getLastStepVehicleIDs('2i')
    vehicles_road3 = traci.edge.getLastStepVehicleIDs('3i')
    vehicles_road4 = traci.edge.getLastStepVehicleIDs('1i')

    # for index in range(4):
    #     edge_index=str(index+1)+'i'
    #     vehicles.append(traci.edge.getLastStepVehicleIDs(edge_index))
    for i in range(12):
        posMatrix.append([])
        velMatrix.append([])
        #         timeMatrix.append([])
        ori_posM.append([])
        ori_velM.append([])
        #         ori_time.append([])
        signalMatrix.append(0)
        for j in range(width):
            posMatrix[i].append(0)
            ori_posM[i].append(0)
            velMatrix[i].append(0)
            ori_velM[i].append(0)
    #             timeMatrix.append(0)
    #             ori_time.append(0)
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
            #             ori_time[traci.vehicle.getLaneIndex(v)][width - 1 - ind] = traci.vehicle.getWaiting

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
    # vm=np.asarray(velMatrix)
    # pm=np.asarray(posMatrix)
    # print(vm.shape,pm.shape)
    # vall=np.stack((vm,pm))
    arrivalrate = '_6666we10'

    stime = time.strftime("%m-%d-%H-%M-%S", time.localtime())
    filename_1 = 'data/output1/vel_' + str(int(PENETRATION_RATE * 100)) + arrivalrate + '_' + stime + '_' + str(
        step) + '.csv'
    vall_f = pd.DataFrame(velMatrix)
    vall_f.to_csv(filename_1, mode='w')

    filename_2 = 'data/output1/pos_' + str(int(PENETRATION_RATE * 100)) + arrivalrate + '_' + stime + '_' + str(
        step) + '.csv'
    vall_f = pd.DataFrame(posMatrix)
    vall_f.to_csv(filename_2, mode='w')

    #
    # svm=np.asarray(ori_velM)
    # spm=np.asarray(ori_posM)
    # print(svm.shape,spm.shape)
    # svall=np.stack((svm,spm))

    filename_3 = 'data/output1/vel' + "S" + str(int(PENETRATION_RATE * 100)) + arrivalrate + '_' + stime + '_' + str(
        step) + '.csv'
    svall_f = pd.DataFrame(ori_velM)
    svall_f.to_csv(filename_3, mode='w')

    filename_4 = 'data/output1/pos' + "S" + str(int(PENETRATION_RATE * 100)) + arrivalrate + '_' + stime + '_' + str(
        step) + '.csv'
    svall_f = pd.DataFrame(ori_posM)
    svall_f.to_csv(filename_4, mode='w')

    filename_5 = 'data/output1/signal' + str(int(PENETRATION_RATE * 100)) + arrivalrate + '_' + stime + '_' + str(
        step) + '.csv'
    svall_f = pd.DataFrame(signalMatrix)
    svall_f.to_csv(filename_5, mode='w')

    filename_6 = 'data/output1/sumo_trainingset.csv'

    with open(filename_6, "a+") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([filename_1, filename_2, filename_3, filename_4, filename_5, PENETRATION_RATE])


# The program looks like this
#    <tlLogic id="0" type="static" programID="0" offset="0">
# the locations of the tls are      NESW
#        <phase duration="31" state="GrGr"/>
#        <phase duration="6"  state="yryr"/>
#        <phase duration="31" state="rGrG"/>
#        <phase duration="6"  state="ryry"/>
#    </tlLogic>

def output_state_v2(step, PENETRATION_RATE):
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

    vehicles_road1 = traci.edge.getLastStepVehicleIDs('4i')
    vehicles_road2 = traci.edge.getLastStepVehicleIDs('2i')
    vehicles_road3 = traci.edge.getLastStepVehicleIDs('3i')
    vehicles_road4 = traci.edge.getLastStepVehicleIDs('1i')

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

    # dirname='data/output2/'+str(currenttime)
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
    # vm=np.asarray(velMatrix)
    # pm=np.asarray(posMatrix)
    # print(vm.shape,pm.shape)
    # vall=np.stack((vm,pm))

    # save data

    arrivalrate = '_6666we10'

    stime = time.strftime("%m-%d-%H-%M-%S", time.localtime())
    filename_1 = 'data/output1/vel_' + str(int(PENETRATION_RATE * 100)) + arrivalrate + '_' + stime + '_' + str(
        step) + '.csv'
    vall_f = pd.DataFrame(velMatrix)
    vall_f.to_csv(filename_1, mode='w')

    filename_2 = 'data/output1/pos_' + str(int(PENETRATION_RATE * 100)) + arrivalrate + '_' + stime + '_' + str(
        step) + '.csv'
    vall_f = pd.DataFrame(posMatrix)
    vall_f.to_csv(filename_2, mode='w')

    filename_3 = 'data/output1/tim_' + str(int(PENETRATION_RATE * 100)) + arrivalrate + '_' + stime + '_' + str(
        step) + '.csv'
    vall_f = pd.DataFrame(timeMatrix)
    vall_f.to_csv(filename_3, mode='w')

    filename_4 = 'data/output1/vel' + "S" + str(int(PENETRATION_RATE * 100)) + arrivalrate + '_' + stime + '_' + str(
        step) + '.csv'
    svall_f = pd.DataFrame(ori_velM)
    svall_f.to_csv(filename_4, mode='w')

    filename_5 = 'data/output1/pos' + "S" + str(int(PENETRATION_RATE * 100)) + arrivalrate + '_' + stime + '_' + str(
        step) + '.csv'
    svall_f = pd.DataFrame(ori_posM)
    svall_f.to_csv(filename_5, mode='w')

    filename_6 = 'data/output1/tim' + "S" + str(int(PENETRATION_RATE * 100)) + arrivalrate + '_' + stime + '_' + str(
        step) + '.csv'
    svall_f = pd.DataFrame(ori_time)
    svall_f.to_csv(filename_6, mode='w')

    filename_7 = 'data/output1/signal' + str(int(PENETRATION_RATE * 100)) + arrivalrate + '_' + stime + '_' + str(
        step) + '.csv'
    svall_f = pd.DataFrame(signalMatrix)
    svall_f.to_csv(filename_7, mode='w')

    filename_8 = 'data/output1/sumo_trainingset.csv'

    with open(filename_8, "a+") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [filename_1, filename_2, filename_3, filename_4, filename_5, filename_6, filename_7, PENETRATION_RATE])


def output_weight_v3(step, PENETRATION_RATE):
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

    vehicles_road1 = traci.edge.getLastStepVehicleIDs('4i')
    vehicles_road2 = traci.edge.getLastStepVehicleIDs('2i')
    vehicles_road3 = traci.edge.getLastStepVehicleIDs('3i')
    vehicles_road4 = traci.edge.getLastStepVehicleIDs('1i')

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
    tposMatrix = torch.tensor(posMatrix, dtype=torch.float64)
    tvelMatrix = torch.tensor(velMatrix, dtype=torch.float64)
    ttimeMatrix = torch.tensor(timeMatrix, dtype=torch.float64)
    tsignalMatrix = torch.tensor(signalMatrix, dtype=torch.float64)

    input1 = torch.stack([tvelMatrix, tposMatrix, ttimeMatrix])
    print input1.size()
    input1 = input1.unsqueeze(0)
    input2 = tsignalMatrix
    input2 = input2.unsqueeze(0)

    model = cnnModel.CNNv2()
    model = model.double()
    model.load_state_dict(torch.load('data/model2_params2_03.pkl'))
    model.eval()
    output = model(input1, input2)
    output = torch.reshape(output, [2, 12, 50])
    print(output[0])
    veh_num = []
    wat_num = []
    for i in range(12):
        veh_num.append(torch.sum(output[0][i][:]))
        wat_num.append(torch.sum(output[0][i][:]))








# def updateweightGetTraci():
#     #weight=[0, 0, 0, 0, 0, 0, 0, 0, 0]
#     volume=[]
#     volume.append(0)
#
#     volume.append(traci.lane.getLastStepVehicleNumber("4i_2"))
#     volume.append(traci.lane.getLastStepVehicleNumber("4i_1"))
#
#     volume.append(traci.lane.getLastStepVehicleNumber("1i_2"))
#     volume.append(traci.lane.getLastStepVehicleNumber("1i_1"))
#
#     volume.append(traci.lane.getLastStepVehicleNumber("3i_2"))
#     volume.append(traci.lane.getLastStepVehicleNumber("3i_1"))
#
#     volume.append(traci.lane.getLastStepVehicleNumber("2i_2"))
#     volume.append(traci.lane.getLastStepVehicleNumber("2i_1"))
#
#     #
#
#
#     print volume
#
#     orlight = 'grrrgrrrgrrrgrrr'
#     orlight = list(orlight)
#
#     trans = {1:[2,3],2:[1],3:[14,15],4:[13],5:[10,11],6:[9],7:[6,7],8:[5]}
#
#     minst=graph.minst(volume)
#
#     print minst
#
#     for i in minst:
#         for j in trans[i]:
#             orlight[j]='G'
#     light=orlight[0]
#     for k in range(1,len(orlight)):
#         light =light + orlight[k]
#
#
#
#
#     print light
#     return light
#
# def transTraci(light,prelight):
#     light=list(light)
#     prelight=list(prelight)
#     translight=''
#     for i in range(len(light)):
#         if light[i]== prelight[i]:
#             translight=translight+light[i]
#         else:
#             if prelight[i]=='G':
#                 translight=translight+'y'
#             else:
#                 translight=translight+light[i]
#
#     return translight
#

def run_collectdata(penetration_rate):


    traci.init(PORT)
    step = 0
    # we start with phase 2 where EW has green
    traci.trafficlights.setPhase("0", 0)
    steps = np.random.randint(3200, size=50)
    steps = steps.tolist()

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        if step in steps:
            output_weight_v3(step, penetration_rate)
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


def run_main(p):
    options = get_options()
    # this script has been called from the command line. It will start sumo as a
    # server, then connect and run
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    # first, generate the route file for this simulation
    # generate_routefile()
    # pWE = 1. /np.random.randint(2,10)
    # pEW = 1. /np.random.randint(2,10)
    # pNS = 1. /np.random.randint(2,10)
    # pSN = 1. /np.random.randint(2,10)
    #
    # pWN = 1. /np.random.randint(2,10)
    # pNE = random.uniform(0, 1)
    # pES = random.uniform(0, 1)
    # pSW = random.uniform(0, 1)

    groute.generate_routefile_v1("va_data/cross.rou.xml")
    # this is the normal way of using traci. sumo is started as a
    # subprocess and then the python script connects and runs
    sumoProcess = subprocess.Popen([sumoBinary, "-c", "va_data/cross.sumocfg", "--summary-output",
                                    "OUTPUT/tripinfo_va.xml", "--remote-port", str(PORT)], stdout=sys.stdout,
                                   stderr=sys.stderr)
    step = run_collectdata(p)


if __name__ == "__main__":
    # results = []
    #
    # options = get_options()
    # # this script has been called from the command line. It will start sumo as a
    # # server, then connect and run
    # if options.nogui:
    #     sumoBinary = checkBinary('sumo')
    # else:
    #     sumoBinary = checkBinary('sumo-gui')
    #
    # # first, generate the route file for this simulation
    # # generate_routefile()
    #
    # # this is the normal way of using traci. sumo is started as a
    # # subprocess and then the python script connects and runs
    # sumoProcess = subprocess.Popen([sumoBinary, "-c", "va_data/cross.sumocfg", "--summary-output",
    #                                 "OUTPUT/tripinfo_va.xml", "--remote-port", str(PORT)], stdout=sys.stdout,
    #                                stderr=sys.stderr)
    # # step = run_collectdata()
    for i in range(10):
        run_main(abs(0.7 + 0.1 * np.random.randn()))

    # run_main(0.7)

# def run():
#     """execute the TraCI control loop"""
#     traci.init(PORT)
#     step = 0
#     # we start with phase 2 where EW has green
#     traci.trafficlights.setPhase("0", 0)
#     curlight = 'gggYgrrrgrrrgrrr'
#     while traci.simulation.getMinExpectedNumber() > 0:
#         # if step%25==0:
#         #     nextlight=updateweightGetTraci()
#         #     #curlight=transTraci(nextlight,curlight) #yellow state
#         # else:
#         #     curlight=nextlight
#         traci.trafficlights.setRedYellowGreenState("0",curlight)
#         #traci.trafficlights.setPhase("0", 0)
#         #updateweight()
#         traci.simulationStep()
#         # output_state(step)
#         # if traci.trafficlights.getPhase("0") == 2:
#         #     if step % 500 > 250:
#         #         traci.trafficlights.setPhase("0", 3)
#         #     else:
#         #         traci.trafficlights.setPhase("0", 2)
#         # we are not already switching
#         # if traci.inductionloop.getLastStepVehicleNumber("0") > 0:
#         #     # there is a vehicle from the north, switch
#         #     traci.trafficlights.setPhase("0", 3)
#         # else:
#         #     # otherwise try to keep green for EW
#         #     traci.trafficlights.setPhase("0", 2)
#         step += 1
#     traci.close()
#     sys.stdout.flush()
#
#
# def get_options():
#     optParser = optparse.OptionParser()
#     optParser.add_option("--nogui", action="store_true",
#                          default=False, help="run the commandline version of sumo")
#     options, args = optParser.parse_args()
#     return options
#
#
# # this is the main entry point of this script
# if __name__ == "__main__":
#     options = get_options()
#
#     # this script has been called from the command line. It will start sumo as a
#     # server, then connect and run
#     if options.nogui:
#         sumoBinary = checkBinary('sumo')
#     else:
#         sumoBinary = checkBinary('sumo-gui')
#
#     # first, generate the route file for this simulation
#     #generate_routefile()
#
#     # this is the normal way of using traci. sumo is started as a
#     # subprocess and then the python script connects and runs
#     sumoProcess = subprocess.Popen([sumoBinary, "-c", "data/cross.sumocfg", "--tripinfo-output",
#                                     "tripinfo.xml", "--remote-port", str(PORT)], stdout=sys.stdout, stderr=sys.stderr)
#     run()
#     sumoProcess.wait()
