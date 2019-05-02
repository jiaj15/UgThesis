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


def generate_routefile_v1(path):
    N = 3200  # number of time steps
    # demand per second from different directions
    # path = "va_data/cross.rou.xml"

    with open(path, "w") as routes:
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
                print >> routes, '    <vehicle id="down_%i" type="SUMO_DEFAULT_TYPE" route="N2S" depart="%i" />' % (
                    vehNr, i)
                vehNr += 1
                lastVeh = i
            if random.uniform(0, 1) < p.pSN:
                print >> routes, '    <vehicle id="up_%i" type="SUMO_DEFAULT_TYPE" route="S2N" depart="%i"/>' % (
                    vehNr, i)
                vehNr += 1
                lastVeh = i

            # ZUO ZHUAN
            if random.uniform(0, 1) < p.pWN:
                print >> routes, '    <vehicle id="rightLT_%i" type="SUMO_DEFAULT_TYPE" route="W2N" depart="%i" color="1,0,0"/>' % (
                    vehNr, i)
                vehNr += 1
                lastVeh = i
            if random.uniform(0, 1) < p.pNE:
                print >> routes, '    <vehicle id="leftLT_%i" type="SUMO_DEFAULT_TYPE" route="N2E" depart="%i" color="1,0,0" />' % (
                    vehNr, i)
                vehNr += 1
                lastVeh = i
            if random.uniform(0, 1) < p.pES:
                print >> routes, '    <vehicle id="downLT_%i" type="SUMO_DEFAULT_TYPE" route="E2S" depart="%i" color="1,0,0"/>' % (
                    vehNr, i)
                vehNr += 1
                lastVeh = i
            if random.uniform(0, 1) < p.pSW:
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


def generate_routefile_v2(PATH, index):
    volume = np.load('npy/luowen_15_oneday.npy')
    arrivalrate = 4 * volume[index][:]
    arrivalrate = arrivalrate / 3600  # +0.15*np.random.randn(12)

    pNE = arrivalrate[0]
    pNS = arrivalrate[1]
    pNW = arrivalrate[2]

    pES = arrivalrate[3]
    pEW = arrivalrate[4]
    pEN = arrivalrate[5]

    pSW = arrivalrate[6]
    pSN = arrivalrate[7]
    pSE = arrivalrate[8]

    pWN = arrivalrate[9]
    pWE = arrivalrate[10]
    pWS = arrivalrate[11]

    N = 2400  # number of time steps
    # demand per second from different directions

    with open(PATH, "w") as routes:
        print >> routes, """<routes>
        <vType id="SUMO_DEFAULT_TYPE" accel="0.8" decel="4.5" sigma="0.5" length="5" minGap="2" maxSpeed="70" color="255,0,0"/>
        <vType id="CAV_TYPE" accel="0.8" decel="4.5" sigma="0.5" length="5" minGap="2" maxSpeed="70" color="176,224,230"/>

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
            if random.uniform(0, 1) < pWE:
                if random.uniform(0, 1) < p.PENETRATION_RATE:
                    type = '    <vehicle id="WE_%i" type="CAV_TYPE" route="W2E" depart="%i" />'
                else:
                    type = '    <vehicle id="WE_%i" type="SUMO_DEFAULT_TYPE" route="W2E" depart="%i" />'
                # print >> routes, '    <vehicle id="right_%i" type="SUMO_DEFAULT_TYPE" route="W2E" depart="%i" />' % (
                #     vehNr, i)
                print >> routes, type % (vehNr, i)
                vehNr += 1
                lastVeh = i
            if random.uniform(0, 1) < pEW:
                if random.uniform(0, 1) < p.PENETRATION_RATE:
                    type = '    <vehicle id="EW_%i" type="CAV_TYPE" route="E2W" depart="%i" />'
                else:
                    type = '    <vehicle id="EW_%i" type="SUMO_DEFAULT_TYPE" route="E2W" depart="%i" />'
                # print >> routes, '    <vehicle id="right_%i" type="SUMO_DEFAULT_TYPE" route="W2E" depart="%i" />' % (
                #     vehNr, i)
                print >> routes, type % (vehNr, i)
                vehNr += 1

                # print >> routes, '    <vehicle id="left_%i" type="SUMO_DEFAULT_TYPE" route="E2W" depart="%i" />' % (
                #     vehNr, i)
                # vehNr += 1
                lastVeh = i
            if random.uniform(0, 1) < pNS:
                if random.uniform(0, 1) < p.PENETRATION_RATE:
                    type = '    <vehicle id="NS_%i" type="CAV_TYPE" route="N2S" depart="%i" />'
                else:
                    type = '    <vehicle id="NS_%i" type="SUMO_DEFAULT_TYPE" route="N2S" depart="%i" />'
                # print >> routes, '    <vehicle id="right_%i" type="SUMO_DEFAULT_TYPE" route="W2E" depart="%i" />' % (
                #     vehNr, i)
                print >> routes, type % (vehNr, i)
                vehNr += 1

                # print >> routes, '    <vehicle id="down_%i" type="SUMO_DEFAULT_TYPE" route="N2S" depart="%i" />' % (
                #     vehNr, i)
                # vehNr += 1
                lastVeh = i
            if random.uniform(0, 1) < pSN:  # 0.01 * np.random.randn() +
                if random.uniform(0, 1) < p.PENETRATION_RATE:
                    type = '    <vehicle id="SN_%i" type="CAV_TYPE" route="S2N" depart="%i" />'
                else:
                    type = '    <vehicle id="SN_%i" type="SUMO_DEFAULT_TYPE" route="S2N" depart="%i" />'
                # print >> routes, '    <vehicle id="right_%i" type="SUMO_DEFAULT_TYPE" route="W2E" depart="%i" />' % (
                #     vehNr, i)
                print >> routes, type % (vehNr, i)
                vehNr += 1
                # print >> routes, '    <vehicle id="up_%i" type="SUMO_DEFAULT_TYPE" route="S2N" depart="%i"/>' % (
                #     vehNr, i)
                # vehNr += 1
                lastVeh = i

            # ZUO ZHUAN
            if random.uniform(0, 1) < pWN:
                if random.uniform(0, 1) < p.PENETRATION_RATE:
                    type = '    <vehicle id="WN_%i" type="CAV_TYPE" route="W2N" depart="%i" />'
                else:
                    type = '    <vehicle id="WN_%i" type="SUMO_DEFAULT_TYPE" route="W2N" depart="%i" />'
                # print >> routes, '    <vehicle id="right_%i" type="SUMO_DEFAULT_TYPE" route="W2E" depart="%i" />' % (
                #     vehNr, i)
                print >> routes, type % (vehNr, i)
                vehNr += 1
                # print >> routes, '    <vehicle id="rightLT_%i" type="SUMO_DEFAULT_TYPE" route="W2N" depart="%i" color="1,0,0"/>' % (
                #     vehNr, i)
                # vehNr += 1
                lastVeh = i
            if random.uniform(0, 1) < pNE:
                if random.uniform(0, 1) < p.PENETRATION_RATE:
                    type = '    <vehicle id="NE_%i" type="CAV_TYPE" route="N2E" depart="%i" />'
                else:
                    type = '    <vehicle id="NE%i" type="SUMO_DEFAULT_TYPE" route="N2E" depart="%i" />'
                # print >> routes, '    <vehicle id="right_%i" type="SUMO_DEFAULT_TYPE" route="W2E" depart="%i" />' % (
                #     vehNr, i)
                print >> routes, type % (vehNr, i)
                vehNr += 1
                # print >> routes, '    <vehicle id="leftLT_%i" type="SUMO_DEFAULT_TYPE" route="N2E" depart="%i" color="1,0,0" />' % (
                #     vehNr, i)
                # vehNr += 1
                lastVeh = i
            if random.uniform(0, 1) < pES:
                if random.uniform(0, 1) < p.PENETRATION_RATE:
                    type = '    <vehicle id="ES_%i" type="CAV_TYPE" route="E2S" depart="%i" />'
                else:
                    type = '    <vehicle id="ES_%i" type="SUMO_DEFAULT_TYPE" route="E2S" depart="%i" />'
                # print >> routes, '    <vehicle id="right_%i" type="SUMO_DEFAULT_TYPE" route="W2E" depart="%i" />' % (
                #     vehNr, i)
                print >> routes, type % (vehNr, i)
                vehNr += 1
                # print >> routes, '    <vehicle id="downLT_%i" type="SUMO_DEFAULT_TYPE" route="E2S" depart="%i" color="1,0,0"/>' % (
                #     vehNr, i)
                # vehNr += 1
                lastVeh = i
            if random.uniform(0, 1) < pSW:
                if random.uniform(0, 1) < p.PENETRATION_RATE:
                    type = '    <vehicle id="SW_%i" type="CAV_TYPE" route="S2W" depart="%i" />'
                else:
                    type = '    <vehicle id="SW_%i" type="SUMO_DEFAULT_TYPE" route="S2W" depart="%i" />'
                # print >> routes, '    <vehicle id="right_%i" type="SUMO_DEFAULT_TYPE" route="W2E" depart="%i" />' % (
                #     vehNr, i)
                print >> routes, type % (vehNr, i)
                vehNr += 1
                # print >> routes, '    <vehicle id="upLT_%i" type="SUMO_DEFAULT_TYPE" route="S2W" depart="%i" color="1,0,0"/>' % (
                #     vehNr, i)
                # vehNr += 1
                lastVeh = i

            # # YOU ZHUAN
            if random.uniform(0, 1) < pWS:
                if random.uniform(0, 1) < p.PENETRATION_RATE:
                    type = '    <vehicle id="WS_%i" type="CAV_TYPE" route="W2S" depart="%i" />'
                else:
                    type = '    <vehicle id="WS_%i" type="SUMO_DEFAULT_TYPE" route="W2S" depart="%i" />'
                # print >> routes, '    <vehicle id="right_%i" type="SUMO_DEFAULT_TYPE" route="W2E" depart="%i" />' % (
                #     vehNr, i)
                print >> routes, type % (vehNr, i)
                vehNr += 1
                # print >> routes, '    <vehicle id="rightRL_%i" type="SUMO_DEFAULT_TYPE" route="W2S" depart="%i" />' % (
                #     vehNr, i)
                # vehNr += 1
                lastVeh = i
            if random.uniform(0, 1) < pSE:
                if random.uniform(0, 1) < p.PENETRATION_RATE:
                    type = '    <vehicle id="SE_%i" type="CAV_TYPE" route="S2E" depart="%i" />'
                else:
                    type = '    <vehicle id="SE_%i" type="SUMO_DEFAULT_TYPE" route="S2E" depart="%i" />'
                # print >> routes, '    <vehicle id="right_%i" type="SUMO_DEFAULT_TYPE" route="W2E" depart="%i" />' % (
                #     vehNr, i)
                print >> routes, type % (vehNr, i)
                vehNr += 1
                # print >> routes, '    <vehicle id="leftRL_%i" type="SUMO_DEFAULT_TYPE" route="S2E" depart="%i" />' % (
                #     vehNr, i)
                # vehNr += 1
                lastVeh = i
            if random.uniform(0, 1) < pEN:
                if random.uniform(0, 1) < p.PENETRATION_RATE:
                    type = '    <vehicle id="EN_%i" type="CAV_TYPE" route="E2N" depart="%i" />'
                else:
                    type = '    <vehicle id="EN_%i" type="SUMO_DEFAULT_TYPE" route="E2N" depart="%i" />'
                # print >> routes, '    <vehicle id="right_%i" type="SUMO_DEFAULT_TYPE" route="W2E" depart="%i" />' % (
                #     vehNr, i)
                print >> routes, type % (vehNr, i)
                vehNr += 1
                # print >> routes, '    <vehicle id="downRL_%i" type="SUMO_DEFAULT_TYPE" route="E2N" depart="%i" color="1,0,0"/>' % (
                #     vehNr, i)
                # vehNr += 1
                lastVeh = i
            if random.uniform(0, 1) < pNW:
                if random.uniform(0, 1) < p.PENETRATION_RATE:
                    type = '    <vehicle id="NW_%i" type="CAV_TYPE" route="N2W" depart="%i" />'
                else:
                    type = '    <vehicle id="NW_%i" type="SUMO_DEFAULT_TYPE" route="N2W" depart="%i" />'
                # print >> routes, '    <vehicle id="right_%i" type="SUMO_DEFAULT_TYPE" route="W2E" depart="%i" />' % (
                #     vehNr, i)
                print >> routes, type % (vehNr, i)
                vehNr += 1
                # print >> routes, '    <vehicle id="upRL_%i" type="SUMO_DEFAULT_TYPE" route="N2W" depart="%i" color="1,0,0"/>' % (
                #     vehNr, i)
                # vehNr += 1
                lastVeh = i

        print >> routes, "</routes>"


def generate_routefile_v4(PATH, indexstart, indexend):
    volume = np.load('npy/luowen_15_oneday.npy')

    N = 900  # number of time steps
    # demand per second from different directions

    with open(PATH, "w") as routes:
        print >> routes, """<routes>
        <vType id="SUMO_DEFAULT_TYPE" accel="0.8" decel="4.5" sigma="0.5" length="5" minGap="2" maxSpeed="70" color="255,0,0"/>
        <vType id="CAV_TYPE" accel="0.8" decel="4.5" sigma="0.5" length="5" minGap="2" maxSpeed="70" color="176,224,230"/>

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
        for index in range(indexstart, indexend):
            arrivalrate = 4 * volume[index][:]
            arrivalrate = arrivalrate / 3600  # +0.15*np.random.randn(12)

            pNE = arrivalrate[0]
            pNS = arrivalrate[1]
            pNW = arrivalrate[2]

            pES = arrivalrate[3]
            pEW = arrivalrate[4]
            pEN = arrivalrate[5]

            pSW = arrivalrate[6]
            pSN = arrivalrate[7]
            pSE = arrivalrate[8]

            pWN = arrivalrate[9]
            pWE = arrivalrate[10]
            pWS = arrivalrate[11]
            for i in range(((index - indexstart) * 900), (index - indexstart + 1) * 900):

                # ZHIXING
                if random.uniform(0, 1) < pWE:
                    if random.uniform(0, 1) < p.PENETRATION_RATE:
                        type = '    <vehicle id="WE_%i" type="CAV_TYPE" route="W2E" depart="%i" />'
                    else:
                        type = '    <vehicle id="WE_%i" type="SUMO_DEFAULT_TYPE" route="W2E" depart="%i" />'
                    # print >> routes, '    <vehicle id="right_%i" type="SUMO_DEFAULT_TYPE" route="W2E" depart="%i" />' % (
                    #     vehNr, i)
                    print >> routes, type % (vehNr, i)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < pEW:
                    if random.uniform(0, 1) < p.PENETRATION_RATE:
                        type = '    <vehicle id="EW_%i" type="CAV_TYPE" route="E2W" depart="%i" />'
                    else:
                        type = '    <vehicle id="EW_%i" type="SUMO_DEFAULT_TYPE" route="E2W" depart="%i" />'
                    # print >> routes, '    <vehicle id="right_%i" type="SUMO_DEFAULT_TYPE" route="W2E" depart="%i" />' % (
                    #     vehNr, i)
                    print >> routes, type % (vehNr, i)
                    vehNr += 1

                    # print >> routes, '    <vehicle id="left_%i" type="SUMO_DEFAULT_TYPE" route="E2W" depart="%i" />' % (
                    #     vehNr, i)
                    # vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < pNS:
                    if random.uniform(0, 1) < p.PENETRATION_RATE:
                        type = '    <vehicle id="NS_%i" type="CAV_TYPE" route="N2S" depart="%i" />'
                    else:
                        type = '    <vehicle id="NS_%i" type="SUMO_DEFAULT_TYPE" route="N2S" depart="%i" />'
                    # print >> routes, '    <vehicle id="right_%i" type="SUMO_DEFAULT_TYPE" route="W2E" depart="%i" />' % (
                    #     vehNr, i)
                    print >> routes, type % (vehNr, i)
                    vehNr += 1

                    # print >> routes, '    <vehicle id="down_%i" type="SUMO_DEFAULT_TYPE" route="N2S" depart="%i" />' % (
                    #     vehNr, i)
                    # vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < pSN:  # 0.01 * np.random.randn() +
                    if random.uniform(0, 1) < p.PENETRATION_RATE:
                        type = '    <vehicle id="SN_%i" type="CAV_TYPE" route="S2N" depart="%i" />'
                    else:
                        type = '    <vehicle id="SN_%i" type="SUMO_DEFAULT_TYPE" route="S2N" depart="%i" />'
                    # print >> routes, '    <vehicle id="right_%i" type="SUMO_DEFAULT_TYPE" route="W2E" depart="%i" />' % (
                    #     vehNr, i)
                    print >> routes, type % (vehNr, i)
                    vehNr += 1
                    # print >> routes, '    <vehicle id="up_%i" type="SUMO_DEFAULT_TYPE" route="S2N" depart="%i"/>' % (
                    #     vehNr, i)
                    # vehNr += 1
                    lastVeh = i

                # ZUO ZHUAN
                if random.uniform(0, 1) < pWN:
                    if random.uniform(0, 1) < p.PENETRATION_RATE:
                        type = '    <vehicle id="WN_%i" type="CAV_TYPE" route="W2N" depart="%i" />'
                    else:
                        type = '    <vehicle id="WN_%i" type="SUMO_DEFAULT_TYPE" route="W2N" depart="%i" />'
                    # print >> routes, '    <vehicle id="right_%i" type="SUMO_DEFAULT_TYPE" route="W2E" depart="%i" />' % (
                    #     vehNr, i)
                    print >> routes, type % (vehNr, i)
                    vehNr += 1
                    # print >> routes, '    <vehicle id="rightLT_%i" type="SUMO_DEFAULT_TYPE" route="W2N" depart="%i" color="1,0,0"/>' % (
                    #     vehNr, i)
                    # vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < pNE:
                    if random.uniform(0, 1) < p.PENETRATION_RATE:
                        type = '    <vehicle id="NE_%i" type="CAV_TYPE" route="N2E" depart="%i" />'
                    else:
                        type = '    <vehicle id="NE%i" type="SUMO_DEFAULT_TYPE" route="N2E" depart="%i" />'
                    # print >> routes, '    <vehicle id="right_%i" type="SUMO_DEFAULT_TYPE" route="W2E" depart="%i" />' % (
                    #     vehNr, i)
                    print >> routes, type % (vehNr, i)
                    vehNr += 1
                    # print >> routes, '    <vehicle id="leftLT_%i" type="SUMO_DEFAULT_TYPE" route="N2E" depart="%i" color="1,0,0" />' % (
                    #     vehNr, i)
                    # vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < pES:
                    if random.uniform(0, 1) < p.PENETRATION_RATE:
                        type = '    <vehicle id="ES_%i" type="CAV_TYPE" route="E2S" depart="%i" />'
                    else:
                        type = '    <vehicle id="ES_%i" type="SUMO_DEFAULT_TYPE" route="E2S" depart="%i" />'
                    # print >> routes, '    <vehicle id="right_%i" type="SUMO_DEFAULT_TYPE" route="W2E" depart="%i" />' % (
                    #     vehNr, i)
                    print >> routes, type % (vehNr, i)
                    vehNr += 1
                    # print >> routes, '    <vehicle id="downLT_%i" type="SUMO_DEFAULT_TYPE" route="E2S" depart="%i" color="1,0,0"/>' % (
                    #     vehNr, i)
                    # vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < pSW:
                    if random.uniform(0, 1) < p.PENETRATION_RATE:
                        type = '    <vehicle id="SW_%i" type="CAV_TYPE" route="S2W" depart="%i" />'
                    else:
                        type = '    <vehicle id="SW_%i" type="SUMO_DEFAULT_TYPE" route="S2W" depart="%i" />'
                    # print >> routes, '    <vehicle id="right_%i" type="SUMO_DEFAULT_TYPE" route="W2E" depart="%i" />' % (
                    #     vehNr, i)
                    print >> routes, type % (vehNr, i)
                    vehNr += 1
                    # print >> routes, '    <vehicle id="upLT_%i" type="SUMO_DEFAULT_TYPE" route="S2W" depart="%i" color="1,0,0"/>' % (
                    #     vehNr, i)
                    # vehNr += 1
                    lastVeh = i

                # # YOU ZHUAN
                if random.uniform(0, 1) < pWS:
                    if random.uniform(0, 1) < p.PENETRATION_RATE:
                        type = '    <vehicle id="WS_%i" type="CAV_TYPE" route="W2S" depart="%i" />'
                    else:
                        type = '    <vehicle id="WS_%i" type="SUMO_DEFAULT_TYPE" route="W2S" depart="%i" />'
                    # print >> routes, '    <vehicle id="right_%i" type="SUMO_DEFAULT_TYPE" route="W2E" depart="%i" />' % (
                    #     vehNr, i)
                    print >> routes, type % (vehNr, i)
                    vehNr += 1
                    # print >> routes, '    <vehicle id="rightRL_%i" type="SUMO_DEFAULT_TYPE" route="W2S" depart="%i" />' % (
                    #     vehNr, i)
                    # vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < pSE:
                    if random.uniform(0, 1) < p.PENETRATION_RATE:
                        type = '    <vehicle id="SE_%i" type="CAV_TYPE" route="S2E" depart="%i" />'
                    else:
                        type = '    <vehicle id="SE_%i" type="SUMO_DEFAULT_TYPE" route="S2E" depart="%i" />'
                    # print >> routes, '    <vehicle id="right_%i" type="SUMO_DEFAULT_TYPE" route="W2E" depart="%i" />' % (
                    #     vehNr, i)
                    print >> routes, type % (vehNr, i)
                    vehNr += 1
                    # print >> routes, '    <vehicle id="leftRL_%i" type="SUMO_DEFAULT_TYPE" route="S2E" depart="%i" />' % (
                    #     vehNr, i)
                    # vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < pEN:
                    if random.uniform(0, 1) < p.PENETRATION_RATE:
                        type = '    <vehicle id="EN_%i" type="CAV_TYPE" route="E2N" depart="%i" />'
                    else:
                        type = '    <vehicle id="EN_%i" type="SUMO_DEFAULT_TYPE" route="E2N" depart="%i" />'
                    # print >> routes, '    <vehicle id="right_%i" type="SUMO_DEFAULT_TYPE" route="W2E" depart="%i" />' % (
                    #     vehNr, i)
                    print >> routes, type % (vehNr, i)
                    vehNr += 1
                    # print >> routes, '    <vehicle id="downRL_%i" type="SUMO_DEFAULT_TYPE" route="E2N" depart="%i" color="1,0,0"/>' % (
                    #     vehNr, i)
                    # vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < pNW:
                    if random.uniform(0, 1) < p.PENETRATION_RATE:
                        type = '    <vehicle id="NW_%i" type="CAV_TYPE" route="N2W" depart="%i" />'
                    else:
                        type = '    <vehicle id="NW_%i" type="SUMO_DEFAULT_TYPE" route="N2W" depart="%i" />'
                    # print >> routes, '    <vehicle id="right_%i" type="SUMO_DEFAULT_TYPE" route="W2E" depart="%i" />' % (
                    #     vehNr, i)
                    print >> routes, type % (vehNr, i)
                    vehNr += 1
                    # print >> routes, '    <vehicle id="upRL_%i" type="SUMO_DEFAULT_TYPE" route="N2W" depart="%i" color="1,0,0"/>' % (
                    #     vehNr, i)
                    # vehNr += 1
                    lastVeh = i

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
            pWE = 0.5 * np.sin(i * 2 * np.pi / 1800)
            pEW = 0.5 * np.sin(i * 2 * np.pi / 1800 + 0.25 * np.pi)
            pNS = 0.5 * np.sin(i * 2 * np.pi / 1800 + 0.5 * np.pi)
            pSN = 0.5 * np.sin(i * 2 * np.pi / 1800 + 0.75 * np.pi)

            # ZHIXING
            if random.uniform(0, 1) < pWE:
                print >> routes, '    <vehicle id="right_%i" type="SUMO_DEFAULT_TYPE" route="W2E" depart="%i" color="1,0,0" />' % (
                    vehNr, i)
                vehNr += 1
                lastVeh = i
            if random.uniform(0, 1) < pEW:
                print >> routes, '    <vehicle id="left_%i" type="SUMO_DEFAULT_TYPE" route="E2W" depart="%i" color="1,0,0" />' % (
                    vehNr, i)
                vehNr += 1
                lastVeh = i
            if random.uniform(0, 1) < pNS:
                print >> routes, '    <vehicle id="down_%i" type="SUMO_DEFAULT_TYPE" route="N2S" depart="%i" color="1,0,0"/>' % (
                    vehNr, i)
                vehNr += 1
                lastVeh = i
            if random.uniform(0, 1) < pSN:
                print >> routes, '    <vehicle id="up_%i" type="SUMO_DEFAULT_TYPE" route="S2N" depart="%i" color="1,0,0"/>' % (
                    vehNr, i)
                vehNr += 1
                lastVeh = i

            # ZUO ZHUAN
            if random.uniform(0, 1) < 0.1 * abs(np.random.randn()):
                print >> routes, '    <vehicle id="rightLT_%i" type="SUMO_DEFAULT_TYPE" route="W2N" depart="%i" color="1,1,1" />' % (
                    vehNr, i)
                vehNr += 1
                lastVeh = i
            if random.uniform(0, 1) < 0.5 * abs(np.random.randn()):
                print >> routes, '    <vehicle id="leftLT_%i" type="SUMO_DEFAULT_TYPE" route="N2E" depart="%i" color="1,1,1"/>' % (
                    vehNr, i)
                vehNr += 1
                lastVeh = i
            if random.uniform(0, 1) < 0.5 * abs(np.random.randn()):
                print >> routes, '    <vehicle id="downLT_%i" type="SUMO_DEFAULT_TYPE" route="E2S" depart="%i" color="1,1,1"/>' % (
                    vehNr, i)
                vehNr += 1
                lastVeh = i
            if random.uniform(0, 1) < 0.1 * abs(np.random.randn()):
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


def generate_routefile_v3(pWE, pEW, pNS, pSN, pWN, pNE, pES, pSW):
    # pWE = random.uniform(0, 1)
    # pEW = random.uniform(0, 1)
    # pNS = random.uniform(0, 1)
    # pSN = random.uniform(0, 1)
    #
    # pWN = random.uniform(0, 1)
    # pNE = random.uniform(0, 1)
    # pES = random.uniform(0, 1)
    # pSW = random.uniform(0, 1)

    N = 3200  # number of time steps
    # demand per second from different directions

    with open("va_data/cross.rou.xml", "w") as routes:
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
            if random.uniform(0, 1) < pWE:
                print >> routes, '    <vehicle id="right_%i" type="SUMO_DEFAULT_TYPE" route="W2E" depart="%i" />' % (
                    vehNr, i)
                vehNr += 1
                lastVeh = i
            if random.uniform(0, 1) < pEW:
                print >> routes, '    <vehicle id="left_%i" type="SUMO_DEFAULT_TYPE" route="E2W" depart="%i" />' % (
                    vehNr, i)
                vehNr += 1
                lastVeh = i
            if random.uniform(0, 1) < pNS:
                print >> routes, '    <vehicle id="down_%i" type="SUMO_DEFAULT_TYPE" route="N2S" depart="%i" />' % (
                    vehNr, i)
                vehNr += 1
                lastVeh = i
            if random.uniform(0, 1) < pSN:
                print >> routes, '    <vehicle id="up_%i" type="SUMO_DEFAULT_TYPE" route="S2N" depart="%i"/>' % (
                    vehNr, i)
                vehNr += 1
                lastVeh = i

            # ZUO ZHUAN
            if random.uniform(0, 1) < pWN:
                print >> routes, '    <vehicle id="rightLT_%i" type="SUMO_DEFAULT_TYPE" route="W2N" depart="%i" color="1,0,0"/>' % (
                    vehNr, i)
                vehNr += 1
                lastVeh = i
            if random.uniform(0, 1) < pNE:
                print >> routes, '    <vehicle id="leftLT_%i" type="SUMO_DEFAULT_TYPE" route="N2E" depart="%i" color="1,0,0" />' % (
                    vehNr, i)
                vehNr += 1
                lastVeh = i
            if random.uniform(0, 1) < pES:
                print >> routes, '    <vehicle id="downLT_%i" type="SUMO_DEFAULT_TYPE" route="E2S" depart="%i" color="1,0,0"/>' % (
                    vehNr, i)
                vehNr += 1
                lastVeh = i
            if random.uniform(0, 1) < pSW:
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



if __name__ == "__main__":
    generate_routefile_v4("va_data/cross.rou.xml", 32, 40)
