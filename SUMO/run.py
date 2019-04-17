import os
import sys
import optparse
import subprocess
import random
import traffic_analysis as ta
import pandas as pd
import PARAMETER as p
import graph
import results as rs
import pandas as pd
import matplotlib.pyplot as plt

random.seed(42)  # make tests reproducible
# we need to import python modules from the $SUMO_HOME/tools directory


results = []

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





def updateWeight(standard):
    global results, PHASERECORD, xsteps, step
    weight = []
    weight.append(0)
    x = locals()
    results_step = []

    for j in p.GRAPH_DIC:
        jroute = p.GRAPH_DIC[j]
        w = 0
        for k in p.lanes_dict[jroute]:
            klane = ta.lane_info(k)
            x[k] = klane

            results_step.append(klane)  # saving results

            if standard == 'w':
                w = w + klane.queuelength

            if standard == 'v':
                w = w + klane.traffic_volume

            if standard == 'vw':
                w = w + klane.AverageWaitingTime + klane.traffic_volume

        weight.append(w)
    results.append(results_step)

    orlight = 'grrrgrrrgrrrgrrr'
    orlight = list(orlight)

    trans = {1: [2, 3], 2: [1], 3: [14, 15], 4: [13], 5: [10, 11], 6: [9], 7: [6, 7], 8: [5]}
    print weight

    minst, PHASEINDEX = graph.MWIS(weight)
    PHASERECORD.append(PHASEINDEX)
    xsteps.append(step)

    print minst

    for i in minst:
        for j in trans[i]:
            orlight[j] = 'g'
    light = orlight[0]
    for k in range(1, len(orlight)):
        light = light + orlight[k]

    print light
    return light


def updatePlusPenalty(standard):
    global g

    global results, PHASERECORD, xsteps
    weight = []
    weight.append(0)
    x = locals()
    results_step = []

    for j in p.GRAPH_DIC:
        jroute = p.GRAPH_DIC[j]
        w = 0
        for k in p.lanes_dict[jroute]:
            klane = ta.lane_info(k)
            x[k] = klane

            results_step.append(klane)  # saving results

            if standard == 'v':
                w = w + klane.AverageWaitingTime

            if standard == 'w':
                w = w + klane.queuelength

            if standard == 'vw':
                w = w + klane.AverageWaitingTime + klane.traffic_volume

        weight.append(w)
    results.append(results_step)

    orlight = 'grrrgrrrgrrrgrrr'
    orlight = list(orlight)

    trans = {1: [2, 3], 2: [1], 3: [14, 15], 4: [13], 5: [10, 11], 6: [9], 7: [6, 7], 8: [5]}
    print weight
    weight2 = g.updateGraphWeight(weight, standard)

    minst, PHASEINDEX = graph.MWIS(weight2)
    PHASERECORD.append(PHASEINDEX)
    xsteps.append(step)
    print step

    g.updateGraphState(minst)

    print minst

    for i in minst:
        for j in trans[i]:
            orlight[j] = 'g'
    light = orlight[0]
    for k in range(1, len(orlight)):
        light = light + orlight[k]

    print light
    return light


def transTraci(light, prelight):
    flag = False
    light = list(light)
    prelight = list(prelight)
    translight = ''
    for i in range(len(light)):
        if light[i] == prelight[i]:
            translight = translight + prelight[i]
        else:
            flag = True
            if prelight[i] == 'g':
                if light[i] == 'r':
                    translight = translight + 'y'
                else:
                    translight = translight + prelight[i]

            else:
                translight = translight + prelight[i]

    return translight, flag


def run(cycle, standard):
    """execute the TraCI control loop"""
    traci.init(PORT)
    global step
    # we start with phase 2 where EW has green
    traci.trafficlights.setPhase("0", 0)
    curlight = 'grrrgrrrgrrrgrrr'
    nextlight = curlight
    flag = False
    while traci.simulation.getMinExpectedNumber() > 0:
        if flag:
            if step % cycle == p.YELLOW_TIME:  # yellow state: 2s
                flag = False
                curlight = nextlight
                traci.trafficlights.setRedYellowGreenState("0", curlight)
            # TRANSFER FROM YELLOW TO NEXTLIGHT

        else:

            if step % cycle == 0:
                # nextlight = updateWeight(standard)
                nextlight = updatePlusPenalty(standard)
                curlight, flag = transTraci(nextlight, curlight)  # yellow state
                traci.trafficlights.setRedYellowGreenState("0", curlight)

        traci.simulationStep()
        step += 1
    traci.close()
    sys.stdout.flush()


def run_noyellow(cycle, standard):
    """execute the TraCI control loop"""
    traci.init(PORT)
    global step
    # we start with phase 2 where EW has green
    traci.trafficlights.setPhase("0", 0)
    curlight = 'grrrgrrrgrrrgrrr'
    nextlight = curlight
    # flag = False
    while traci.simulation.getMinExpectedNumber() > 0:

        if step % cycle == 0:
            nextlight = updateWeight(standard)

            traci.trafficlights.setRedYellowGreenState("0", nextlight)


        traci.simulationStep()
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

    # this is the normal way of using traci. sumo is started as a
    # subprocess and then the python script connects and runs
    # sumoProcess = subprocess.Popen([sumoBinary, "-c", "data/cross.sumocfg", "--tripinfo-output",
    #                                 "tripinfo.xml", "--remote-port", str(PORT)], stdout=sys.stdout, stderr=sys.stderr)

    cycle = [5]
    stand = ['w']
    # graph=[]
    for i in cycle:
        for j in stand:
            PHASERECORD = []
            xsteps = []
            step = 0
            results = []

            g = graph.Graph(i)
            g.updateGraphState([])
            # for index in range(1,9):
            #     graph.append(graph.Vertices(index,i))

            options = get_options()
            name = str(j) + str(i) + '.xml'

            if options.nogui:
                sumoBinary = checkBinary('sumo')
            else:
                sumoBinary = checkBinary('sumo-gui')
            sumoProcess = subprocess.Popen([sumoBinary, "-c", "data/cross.sumocfg", "--tripinfo-output",
                                            "OUTPUT/" + name, "--remote-port", str(PORT)], stdout=sys.stdout,
                                           stderr=sys.stderr)

            run(i, j)

            # sumoProcess.wait()
            df = rs.SaveResults(results, i, j)

            results = []

            y = PHASERECORD
            x = xsteps
            data_f = pd.DataFrame(x, y)
            filename_da = "OUTPUT_2/" + name + "_phaseRecord.csv"
            data_f.to_csv(filename_da, mode="w+")
            plt.plot(x, y)
            plt.show()
