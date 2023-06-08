
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd

class Geometry:
    """
    This class is defined to get the geometry of the mesh

    """
    def __init__(self, PD):
        self.PD = PD
    def drawmesh(self):

        EL = np.loadtxt("src/data/ELT.txt", dtype=int)
        NL = np.loadtxt("src/data/NLT.txt", dtype=float)

        return (NL, EL)
