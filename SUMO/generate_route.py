import os
import sys
import random
import PARAMETER as p

import numpy as np

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


def generate_routefile():

    N = 3200 # number of time steps
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
            if random.uniform(0, 1) < 0.01 * np.random.randn() + p.pWE:
                print >> routes, '    <vehicle id="right_%i" type="SUMO_DEFAULT_TYPE" route="W2E" depart="%i" />' % (
                    vehNr, i)
                vehNr += 1
                lastVeh = i
            if random.uniform(0, 1) < 0.01 * np.random.randn() + p.pEW:
                print >> routes, '    <vehicle id="left_%i" type="SUMO_DEFAULT_TYPE" route="E2W" depart="%i" />' % (
                    vehNr, i)
                vehNr += 1
                lastVeh = i
            if random.uniform(0, 1) < 0.01 * np.random.randn() + p.pNS:
                print >> routes, '    <vehicle id="down_%i" type="SUMO_DEFAULT_TYPE" route="N2S" depart="%i" />' % (
                    vehNr, i)
                vehNr += 1
                lastVeh = i
            if random.uniform(0, 1) < 0.01 * np.random.randn() + p.pSN:
                print >> routes, '    <vehicle id="up_%i" type="SUMO_DEFAULT_TYPE" route="S2N" depart="%i"/>' % (
                    vehNr, i)
                vehNr += 1
                lastVeh = i

            # ZUO ZHUAN
            if random.uniform(0, 1) < 0.01 * np.random.randn() + p.pWN:
                print >> routes, '    <vehicle id="rightLT_%i" type="SUMO_DEFAULT_TYPE" route="W2N" depart="%i" color="1,0,0"/>' % (
                    vehNr, i)
                vehNr += 1
                lastVeh = i
            if random.uniform(0, 1) < 0.01 * np.random.randn() + p.pNE:
                print >> routes, '    <vehicle id="leftLT_%i" type="SUMO_DEFAULT_TYPE" route="N2E" depart="%i" color="1,0,0" />' % (
                    vehNr, i)
                vehNr += 1
                lastVeh = i
            if random.uniform(0, 1) < 0.05 * np.random.randn() + p.pES:
                print >> routes, '    <vehicle id="downLT_%i" type="SUMO_DEFAULT_TYPE" route="E2S" depart="%i" color="1,0,0"/>' % (
                    vehNr, i)
                vehNr += 1
                lastVeh = i
            if random.uniform(0, 1) < 0.05 * np.random.randn() + p.pSW:
                print >> routes, '    <vehicle id="upLT_%i" type="SUMO_DEFAULT_TYPE" route="S2W" depart="%i" color="1,0,0"/>' % (
                    vehNr, i)
                vehNr += 1
                lastVeh = i

            # # YOU ZHUAN
            # if random.uniform(0, 1) < p.pWS:
            #     print >> routes, '    <vehicle id="rightRL_%i" type="SUMO_DEFAULT_TYPE" route="W2S" depart="%i" />' % (
            #         vehNr, i)
            #     vehNr += 1
            #     lastVeh = i
            # if random.uniform(0, 1) < p.pSE:
            #     print >> routes, '    <vehicle id="leftRL_%i" type="SUMO_DEFAULT_TYPE" route="S2E" depart="%i" />' % (
            #         vehNr, i)
            #     vehNr += 1
            #     lastVeh = i
            # if random.uniform(0, 1) < p.pEN:
            #     print >> routes, '    <vehicle id="downRL_%i" type="SUMO_DEFAULT_TYPE" route="E2N" depart="%i" color="1,0,0"/>' % (
            #         vehNr, i)
            #     vehNr += 1
            #     lastVeh = i
            # if random.uniform(0, 1) < p.pNW:
            #     print >> routes, '    <vehicle id="upRL_%i" type="SUMO_DEFAULT_TYPE" route="N2W" depart="%i" color="1,0,0"/>' % (
            #         vehNr, i)
            #     vehNr += 1
            #     lastVeh = i

        print >> routes, "</routes>"


def generate_routefile_vari():
    N = 3200  # number of time steps
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
            if random.uniform(0, 1) < abs(np.random.randn()):
                print >> routes, '    <vehicle id="right_%i" type="SUMO_DEFAULT_TYPE" route="W2E" depart="%i" color="1,0,0" />' % (
                    vehNr, i)
                vehNr += 1
                lastVeh = i
            if random.uniform(0, 1) < abs(np.random.randn()):
                print >> routes, '    <vehicle id="left_%i" type="SUMO_DEFAULT_TYPE" route="E2W" depart="%i" color="1,0,0" />' % (
                    vehNr, i)
                vehNr += 1
                lastVeh = i
            if random.uniform(0, 1) < abs(np.random.randn()):
                print >> routes, '    <vehicle id="down_%i" type="SUMO_DEFAULT_TYPE" route="N2S" depart="%i" color="1,0,0"/>' % (
                    vehNr, i)
                vehNr += 1
                lastVeh = i
            if random.uniform(0, 1) < abs(np.random.randn()):
                print >> routes, '    <vehicle id="up_%i" type="SUMO_DEFAULT_TYPE" route="S2N" depart="%i" color="1,0,0"/>' % (
                    vehNr, i)
                vehNr += 1
                lastVeh = i

            # ZUO ZHUAN
            if random.uniform(0, 1) < abs(np.random.randn()):
                print >> routes, '    <vehicle id="rightLT_%i" type="SUMO_DEFAULT_TYPE" route="W2N" depart="%i" color="1,1,1" />' % (
                    vehNr, i)
                vehNr += 1
                lastVeh = i
            if random.uniform(0, 1) < abs(np.random.randn()):
                print >> routes, '    <vehicle id="leftLT_%i" type="SUMO_DEFAULT_TYPE" route="N2E" depart="%i" color="1,1,1"/>' % (
                    vehNr, i)
                vehNr += 1
                lastVeh = i
            if random.uniform(0, 1) < abs(np.random.randn()):
                print >> routes, '    <vehicle id="downLT_%i" type="SUMO_DEFAULT_TYPE" route="E2S" depart="%i" color="1,1,1"/>' % (
                    vehNr, i)
                vehNr += 1
                lastVeh = i
            if random.uniform(0, 1) < abs(np.random.randn()):
                print >> routes, '    <vehicle id="upLT_%i" type="SUMO_DEFAULT_TYPE" route="S2W" depart="%i" color="1,1,1"/>' % (
                    vehNr, i)
                vehNr += 1
                lastVeh = i

            # # YOU ZHUAN
            # if random.uniform(0, 1) < p.pWS:
            #     print >> routes, '    <vehicle id="rightRL_%i" type="SUMO_DEFAULT_TYPE" route="W2S" depart="%i" />' % (
            #         vehNr, i)
            #     vehNr += 1
            #     lastVeh = i
            # if random.uniform(0, 1) < p.pSE:
            #     print >> routes, '    <vehicle id="leftRL_%i" type="SUMO_DEFAULT_TYPE" route="S2E" depart="%i" />' % (
            #         vehNr, i)
            #     vehNr += 1
            #     lastVeh = i
            # if random.uniform(0, 1) < p.pEN:
            #     print >> routes, '    <vehicle id="downRL_%i" type="SUMO_DEFAULT_TYPE" route="E2N" depart="%i" color="1,0,0"/>' % (
            #         vehNr, i)
            #     vehNr += 1
            #     lastVeh = i
            # if random.uniform(0, 1) < p.pNW:
            #     print >> routes, '    <vehicle id="upRL_%i" type="SUMO_DEFAULT_TYPE" route="N2W" depart="%i" color="1,0,0"/>' % (
            #         vehNr, i)
            #     vehNr += 1
            #     lastVeh = i

        print >> routes, "</routes>"


if __name__ == "__main__":
    generate_routefile()
