import numpy as np
import math

class disp_force_assemble:
    def __init__(self, NoN, PD):
        self.NoN = NoN
        self.PD = PD

    def assemble_displacement(self, ENL, NL):              #here I defined the prescribed displacement in my case, nodes that have prescribed displacements
        self.NoN = np.size(NL, 0)
        self.PD = np.size(NL, 1)
        DOC = 0
        Up = []
        for i in range(0, self.NoN):
            for j in range(0, self.PD):
                if ENL[i, self.PD+j] == -1:
                    DOC +=1
                    Up.append(ENL[i, 4*self.PD+j])
        Up = np.vstack([Up]).reshape(-1,1)

        return Up


    def assemble_forces(self, ENL, NL):             #here I defined the prescribed nodal forces in my case, nodes that have prescribed forces applied on them
        self.NoN = np.size(NL, 0)
        self.PD = np.size(NL, 1)
        DOF = 0
        Fp = []

        for i in range(0, self.NoN):
            for j in range(0, self.PD):
                if ENL[i, self.PD+j] == 1:
                    DOF +=1
                    Fp.append(ENL[i, 5*self.PD+j])

        Fp = np.vstack([Fp]).reshape(-1,1)

        return Fp

    def updatenodes(self, ENL, Uu, Fu, NL):            #here I put the obtained unknown displacement and nodal forces(in Runnerfile) into ENL
        self.NoN = np.size(NL, 0)
        self.PD = np.size(NL, 1)
        DOFS = 0
        DOCS = 0

        for i in range(0, self.NoN):
            for j in range(0, self.PD):
                if ENL[i, self.PD+j] == 1:
                    DOFS += 1
                    ENL[i, 4*self.PD+j] = Uu[DOFS-1]
                else:
                    DOCS += 1
                    ENL[i, 5*self.PD+j] = Fu[DOCS-1]


        return ENL
