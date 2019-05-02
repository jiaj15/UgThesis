#!/usr/bin/env python

import os
import sys
import optparse
import subprocess


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

import PARAMETER as p

PORT = 8873


class vehicle_info:
    ID = ''
    waitingtime = 0

    def __init__(self, ID, w):
        self.vehicle_ID = ID
        self.waitingtime = w


class lane_info:
    ID = ''
    # lane=[]
    AverageWaitingTime = 0
    traffic_volume = 0
    queuelength=0
    # vehicle_num=0
    # vehicles_lane_waitingtime = 0

    def __init__(self, ID):
        self.ID = ID
        vehicles_lane_waitingtime = traci.lane.getWaitingTime(self.ID)
        self.queuelength = traci.lane.getLastStepHaltingNumber(self.ID)
        # print self.queuelength
        # self.traffic_volume = traci.inductionloop.getLastStepVehicleNumber(self.ID)
        self.traffic_volume = traci.lane.getLastStepVehicleNumber(self.ID)
        # num=self.queuelength
        # self.vehicle_num=traci.lane.getLastStepVehicleNumber(self.ID)
        num = traci.lane.getLastStepHaltingNumber(self.ID)
        vehicles = traci.lane.getLastStepVehicleIDs(self.ID)

        if num is not 0:
            self.AverageWaitingTime = (vehicles_lane_waitingtime + 2 * traci.vehicle.getWaitingTime(vehicles[0])) / num
        # sum=0
        #
        # for index in range(len(self.vehicles_lane_waitingtime)):
        #     sum = sum + self.vehicles_lane_waitingtime[index]
        # self.AverageWaitingTime = sum / len(self.vehicles_lane_waitingtime)

    # def getAverageWaitingTime(self):
    #     sum = 0
    #     for vehicle in self.vehicles_lane_waitingtime:
    #         sum = sum + self.vehicles_lane_waitingtime[vehicle]
    #     self.AverageWaitingTime = sum / len(self.vehicles_lane_waitingtime)
    #

# class steplistener:
#     step = 0
#     intersection_volume = {}
#     intersection_traveltime = {}
#
#     def __init__(self,step):
#         self.step=step
#         self.intersection_traveltime[step]=0




def getlanes_info():
    x = locals()
    for lanes in p.routes:
        for lane in p.lanes_dict[lanes]:
            ilane = lane_info(lane)
            # ilane.getTrafficVolume()
            # vehicles = traci.lane.getLastStepVehicleIDs(lane)
            # for vehicle in vehicles:
            #     ilane.vehicles_lane_waitingtime[vehicle] = traci.vehicle.getAccumulatedWaitingTime(vehicle)
            # ilane.getAverageWaitingTime()
            x[lane] = ilane
    return x

