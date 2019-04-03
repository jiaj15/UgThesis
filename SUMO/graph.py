#!/usr/bin/env python

import numpy as np
import PARAMETER as p

#global variables
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
conflict=[[3,6,7,8],[3,4,5,8],[1,2,5,8],[2,5,6,7],[2,3,4,7],[1,4,7,8],[1,4,5,6],[1,2,3,6]]
# nonconflict=[[1,2,4,5],[1,2,6,7],[3,4,6,7],[1,3,4,8],[1,5,6,8],[2,3,5,6],[2,3,7,8],[4,5,7,8]]
nonconflict=[[1,2,4,5],[2,6,7],[3,4,6,7],[4,8],[5,6,8],[6],[7,8],[8]]

#weight=[0,0, 0, 0, 0, 0, 0, 0, 0]


        


def minst(weight):
    n=8

    IS=[]
    IS_w=[]
    #independent set

    for j in range(n):
        flag=[0,0,0,0,0,0,0,0,0]
        # flag[j]=1
        tmp=[]
        tmp_w=0

        for ik in nonconflict[j]:

            for k in range(len(tmp)):
                if ik not in nonconflict[tmp[k]]:
                    flag[ik]=2
                    break
            if flag[ik] ==0:
                tmp.append(ik)
                tmp_w =tmp_w + weight[ik]
        IS.append(tmp)
        IS_w.append(tmp_w)
    # maximal weight
    maxW=0
    max=0
    print IS
    print IS_w
    for i in range(len(IS_w)):

        if IS_w[i]> maxW:
            maxW=IS_w[i]
            max=i
#    print IS[max]
    return IS[max]

# if __name__ == "__main__":
#     weight=[0,7, 6, 5, 4, 3, 2, 1, 0]
#     minst(weight)











