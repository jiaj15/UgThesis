import os
import sys

import optparse
import subprocess
import random
import run
import matplotlib.pyplot as plt
import pandas

phasesrecord = []
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


def record_queuelength():
    global results
    result = 0
    result = result + traci.edge.getLastStepHaltingNumber('1i')
    result = result + traci.edge.getLastStepHaltingNumber('2i')
    result = result + traci.edge.getLastStepHaltingNumber('3i')
    result = result + traci.edge.getLastStepHaltingNumber('4i')

    results.append(result)
def run_bench():
    global phasesrecord

    traci.init(PORT)
    step = 0
    # we start with phase 2 where EW has green
    traci.trafficlights.setPhase("0", 0)

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        phasesrecord.append(traci.trafficlights.getPhase("0"))
        record_queuelength()


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


if __name__ == "__main__":
    results = []

    options = get_options()
    # this script has been called from the command line. It will start sumo as a
    # server, then connect and run
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    # first, generate the route file for this simulation
    # generate_routefile()

    # this is the normal way of using traci. sumo is started as a
    # subprocess and then the python script connects and runs
    sumoProcess = subprocess.Popen([sumoBinary, "-c", "va_data/cross.sumocfg", "--summary-output",
                                    "OUTPUT/tripinfo_va.xml", "--remote-port", str(PORT)], stdout=sys.stdout,
                                   stderr=sys.stderr)
    step = run_bench()
    x = range(step)
    y = []
    for phase in phasesrecord:
        y.append(int(phase))
    # y=phasesrecord

    data_f = pandas.DataFrame(y, x)
    filename_da = "OUTPUT/benchmark_phaseRecord.csv"
    data_f.to_csv(filename_da, mode="w+")
    plt.plot(x, y)
    plt.show()

    data_j = pandas.DataFrame({"queue length": results, "steps": x})
    filename_ql = "data/output/31-r.csv"
    data_j.to_csv(filename_ql, mode="w+")
