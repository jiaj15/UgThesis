import os
import sys

import optparse
import subprocess
import random
import run
import results as rs

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


def run_bench():
    global results
    traci.init(PORT)
    step = 0
    # we start with phase 2 where EW has green
    traci.trafficlights.setPhase("0", 0)

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        print traci.trafficlights.getPhase("0")
        # run.updateWeight('w')

        step += 1
    traci.close()
    sys.stdout.flush()


def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = optParser.parse_args()
    return options


if __name__ == "__main__":

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
                                    "tripinfo_va.xml", "--remote-port", str(PORT)], stdout=sys.stdout,
                                   stderr=sys.stderr)
    run_bench()
    rs.SaveResults(results, 90, 'v')
