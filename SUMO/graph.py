#!/usr/bin/env python

import numpy as np
import PARAMETER as p

# global variables
Graph = [
    [0, 0, 1, 0, 0, 1, 1, 1],
    [0, 0, 1, 1, 1, 0, 0, 1],
    [1, 1, 0, 0, 1, 0, 0, 1],
    [0, 1, 0, 0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0, 0, 1, 0],
    [1, 0, 0, 1, 0, 0, 1, 1],
    [1, 0, 0, 1, 1, 1, 0, 0],
    [1, 1, 1, 0, 0, 1, 0, 0]
]
conflict = [[], [3, 6, 7, 8], [3, 4, 5, 8], [1, 2, 5, 8], [2, 5, 6, 7], [2, 3, 4, 7], [1, 4, 7, 8], [1, 4, 5, 6],
            [1, 2, 3, 6]]
nonconflict = [[], [1, 2, 4, 5], [1, 2, 6, 7], [3, 4, 6, 7], [1, 3, 4, 8], [1, 5, 6, 8], [2, 3, 5, 6], [2, 3, 7, 8],
               [4, 5, 7, 8]]


# nonconflict = [[],[1, 2, 4, 5], [2, 6, 7], [3, 4, 6, 7], [4, 8], [5, 6, 8], [6], [7, 8], [8]]
# NCV=[[1, 2], [1, 4], [1, 5], [2, 6], [2, 7], [3, 4], [3, 6], [3, 7], [4, 8], [5, 6], [5, 8], [7, 8]]

# weight=[0,0, 0, 0, 0, 0, 0, 0, 0]


class Vertices:
    ID = 0
    Gtime = 0
    curState = ''
    weight = 0
    T = 0

    def __init__(self, id, cycle):
        self.ID = id
        self.T = cycle

    def updateState(self, cur):
        self.curState = cur
        if cur == 'G':
            self.Gtime = self.Gtime + self.T
        else:
            self.Gtime = 0

    def updateWeight(self, weight, standard):
        self.weight = weight[self.ID]
        if self.curState == 'G':
            if self.Gtime < 45:
                if self.weight is not 0:
                    if standard == 'v':
                        self.weight = + (400 - self.Gtime)
                    else:
                        self.weight = + (400 - self.Gtime)


class Graph():
    graph = []

    def __init__(self, cycle):
        self.graph.append(0)
        for i in range(1, 9):
            self.graph.append(Vertices(i, cycle))

    def updateGraphState(self, MWIS):
        for i in range(1, 9):
            if i in MWIS:
                self.graph[i].updateState('G')
            else:
                self.graph[i].updateState('r')

    def updateGraphWeight(self, weight, standard):
        for i in range(1, 9):
            self.graph[i].updateWeight(weight, standard)
            weight[i] = self.graph[i].weight
        return weight





def MIS():
    IS = []
    for i in range(1, 9):
        for j in range(1, i):

            if j in nonconflict[i]:
                k = set(nonconflict[i]) & set(nonconflict[j])
                IS.append(k)

    print IS
    return IS


def MWIS(weight):
    print weight
    IS = [set([1, 2]), set([1, 4]), set([3, 4]), set([1, 5]), set([2, 6]), set([3, 6]), set([5, 6]), set([2, 7]),
          set([3, 7]), set([8, 4]), set([8, 5]), set([8, 7])]

    W = 0
    # W=np.zeros(len(IS))
    MAX = 0
    MAXIndex = -1
    for k in range(len(IS)):
        W = 0
        for j in IS[k]:
            # print j
            W = W + weight[j]
            # print 'update weight'

        if W >= MAX:
            MAX = W
            MAXIndex = k
    # print MAXIndex
    # print W
    print IS[MAXIndex]
    return IS[MAXIndex], MAXIndex

#
# if __name__ == "__main__":
#     weight = [0, 16.0, 0, 12038.0, 2.0, 0, 3017.0, 0, 2437.0]
#     MIS()
#     MWIS(weight)
