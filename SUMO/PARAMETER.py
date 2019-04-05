# probability of approaching vehicles

pWE = 1. / 3
pEW = 1. / 3
pNS = 1. / 30
pSN = 1. / 30

pWN = 1. / 30
pNE = 1. / 30
pES = 1. / 30
pSW = 1. / 30

# pWS = 1. / 20
# pSE = 1. / 10
# pEN = 1. / 20
# pNW = 1. / 10

YELLOW_TIME = 2
PENETRATION_RATE = 9. / 10
WIDTH = 70

# <vType id="SUMO_DEFAULT_TYPE"
# accel="0.8" decel="4.5" sigma="0" length="5" minGap="2" maxSpeed="70"/>

routes = ['W2E', 'E2W', 'N2S', 'S2N', 'W2N', 'N2E', 'E2S', 'S2W', 'W2S', 'S2E', 'E2N', 'N2W']
# lanes_dict = {'W2E': ['1i_0', '1i_1'], 'E2W': ['2i_0', '2i_1'], 'N2S': ['4i_0', '4i_1'], 'S2N': ['3i_0', '3i_1'],
#               'W2N': ['1i_2'], 'N2E': ['4i_2'], 'E2S': ['2i_2'], 'S2W': ['3i_2']}

lanes_dict = {'W2E': ['1i_1'], 'E2W': [ '2i_1'], 'N2S': [ '4i_1'], 'S2N': ['3i_1'],
              'W2N': ['1i_2'], 'N2E': ['4i_2'], 'E2S': ['2i_2'], 'S2W': ['3i_2']}




GRAPH_DIC = {1: 'N2E', 2: 'N2S', 3: 'W2N', 4: 'W2E', 5: 'S2W', 6: 'S2N', 7: 'E2S', 8: 'E2W'}
trl_lanes_dict = {}
# parameter of input matrix

CELL_LENGTH = 7
OFFSET = 11
SPEEDLIMIT = 14  #

# CNN

EPOCH = 1
BATCH_SIZE = 50
LR = 0.3
