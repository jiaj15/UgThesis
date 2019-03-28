# probability of approaching vehicles

pWE = 1. / 10
pEW = 1. / 10
pNS = 1. / 10
pSN = 1. / 10

pWN = 1. / 100
pNE = 1. / 100
pES = 1. / 100
pSW = 1. / 100

pWS = 1. / 10
pSE = 1. / 10
pEN = 1. / 10
pNW = 1. / 10

PENETRATION_RATE = 9. / 10
WIDTH = 70

# <vType id="SUMO_DEFAULT_TYPE"
# accel="0.8" decel="4.5" sigma="0" length="5" minGap="2" maxSpeed="70"/>

routes = ['W2E', 'E2W', 'N2S', 'S2N', 'W2N', 'N2E', 'E2S', 'S2W', 'W2S', 'S2E', 'E2N', 'N2W']
# parameter of input matrix

CELL_LENGTH = 7
OFFSET = 11
SPEEDLIMIT = 14  #

# CNN

EPOCH = 1
BATCH_SIZE = 50
LR = 0.3
