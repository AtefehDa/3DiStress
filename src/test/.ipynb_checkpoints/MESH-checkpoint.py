
import numpy as np
import matplotlib.pyplot as plt
import math


class Geometry:
    """
    This class is defined to get the geometry of the mesh

    """
    def __init__(self, d1, d2, p, m, PD):
        self.d1 = d1
        self.d2 = d2
        self.p = p
        self.m = m
        self.PD = PD

    def drawmesh(self):

        q = np.array ([[0,0], [self.d1, 0], [0, self.d2], [self.d1, self.d2]])
        NoN = (self.p + 1) * (self.m + 1)
        NoE = self.p * self.m
        NPE = 4
        a = (q[1, 0] - q[0, 0]) / self.p
        b = (q[2, 1] - q[0, 1]) / self.m

        NL = np.zeros([NoN, self.PD])
        n = 0
        for i in range(1, self.m+2):
            for j in range(1, self.p+2):
                NL[n, 0] = q[0,0] + (j-1)*a
                NL[n, 1] = q[0,1] + (i-1)*b
                n += 1
        EL = np.zeros([NoE, NPE])
        for i in range(1, self.m+1):
            for j in range(1, self.p+1):
                if j==1 :
                    EL[(i-1) * self.p + j-1, 0] = (i-1)*(self.p+1) + j
                    EL[(i-1) * self.p + j-1, 1] = EL[(i-1)*self.p + j-1, 0] + 1
                    EL[(i-1) * self.p + j-1, 3] = EL[(i-1)*self.p + j-1, 0] + (self.p+1)
                    EL[(i-1) * self.p + j-1, 2] = EL[(i-1)*self.p + j-1, 3] + 1
                else:
                    EL[(i-1) * self.p + j-1, 0] = EL[(i-1) * self.p + j-2, 1]
                    EL[(i-1) * self.p + j-1, 3] = EL[(i-1) * self.p + j-2, 2]
                    EL[(i-1) * self.p + j-1, 1] = EL[(i-1) * self.p + j-1, 0] + 1
                    EL[(i-1) * self.p + j-1, 2] = EL[(i-1) * self.p + j-1, 3] + 1

        return (NL, EL)