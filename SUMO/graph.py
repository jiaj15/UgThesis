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


# def minst(weight):
#     n = 8
#
#     IS = []
#     IS_w = []
#     # independent set
#
#     for j in range(n):
#         flag = [0, 0, 0, 0, 0, 0, 0, 0, 0]
#         # flag[j]=1
#         tmp = []
#         tmp_w = 0
#
#         for ik in nonconflict[j]:
#
#             for k in range(len(tmp)):
#                 if ik not in nonconflict[tmp[k]]:
#                     flag[ik] = 2
#                     break
#             if flag[ik] == 0:
#                 tmp.append(ik)
#                 tmp_w = tmp_w + weight[ik]
#         IS.append(tmp)
#         IS_w.append(tmp_w)
#     # maximal weight
#     maxW = 0
#     max = 0
#     print IS
#     print IS_w
#     for i in range(len(IS_w)):
#
#         if IS_w[i] > maxW:
#             maxW = IS_w[i]
#             max = i
#     #    print IS[max]
#     return IS[max]
# def geNumber():
#     IK=[]
#     for i in range(8):
#         for j in nonconflict[i]:
#             if i+1 !=j:
#                 IK.append([i+1,j])
#
#     return IK
#


# def minst_v2(weight):
#
#     n = 8
#     M=np.zeros([9,9])
#     for k in range(9):
#         M[k][k]=weight[k]
#
#     for j in range(1,n+1):
#         ik =set(range(1,9)).difference(set(conflict[j]))
#         for i in range(1,j):
#             for element in ik:
#
#
#                 if element in range(i,j) and M[i][j-1]<M[i][element-1]+weight[element]+M[element+1][j-1]:
#                     M[i][j]=M[i][element-1]+weight[element]+M[element+1][j-1]
#                 else:
#                     M[i][j] = M[i][j - 1]
#     return M
#

def MIS():
    IS=[]
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
    return IS[MAXIndex]


if __name__ == "__main__":
    weight = [0, 16.0, 0, 12038.0, 2.0, 0, 3017.0, 0, 2437.0]
    MIS()
    MWIS(weight)
