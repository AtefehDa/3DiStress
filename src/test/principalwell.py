import math

import numpy as np
import pandas as pd
import scipy

class principal:

    def __init__(self, PD, NoN, NoE, NPE, GPE):
        self.PD = PD
        self.NoN = NoN
        self.NoE = NoE
        self.NPE = NPE
        self.GPE = GPE

    def location(self, NL_Cartesian, EL):
        nodes_well1 = []
        nodes_well2 = []
        nodes_well3 = []
        nodes_well4 = []
        nodes_well22b = []
        nodes_well33b = []
        nodes_well44b = []

        for i in range(0, self.NoN):                                             #i is python index

            if  1 < math.sqrt((NL_Cartesian[i, 0]-419.87)**2 + (NL_Cartesian[i, 1]-1240.94)**2) < 5:    #ccor well1(EAW6) = (419.87, 1240.94)
                r = math.sqrt((NL_Cartesian[i, 0]-419.87)**2 + (NL_Cartesian[i, 1]-1240.94)**2)
                if 40 < NL_Cartesian[i, 2]:
                    nodes_well1.append((i, r))

            if 1 < math.sqrt((NL_Cartesian[i, 0]-1001.67)**2 + (NL_Cartesian[i, 1]-865.13)**2) < 5:      #ccor well2(ED17) = (1001.67, 865.13)
                r = math.sqrt((NL_Cartesian[i, 0]-1001.67)**2 + (NL_Cartesian[i, 1]-865.13)**2)
                if 42 < NL_Cartesian[i, 2]:
                    nodes_well2.append((i, r))
                if NL_Cartesian[i, 2] < 32:
                    nodes_well22b.append((i, r))

            if 1 < math.sqrt((NL_Cartesian[i, 0]-288.83)**2 + (NL_Cartesian[i, 1]-368.27)**2) < 10:      #ccor well3(ED18) = (288.83, 368.27)
                r = math.sqrt((NL_Cartesian[i, 0]-288.83)**2 + (NL_Cartesian[i, 1]-368.27)**2)
                if 25 < NL_Cartesian[i, 2] :
                    nodes_well3.append((i,r))
                if NL_Cartesian[i, 2] < 15:
                    nodes_well33b.append((i, r))

            if 1 < math.sqrt((NL_Cartesian[i, 0]-920.45)**2 + (NL_Cartesian[i, 1]-270.6)**2) < 10:       #ccor well4(ED35) = (920.45, 270.6)
                r = math.sqrt((NL_Cartesian[i, 0]-920.45)**2 + (NL_Cartesian[i, 1]-270.6)**2)
                if 40 < NL_Cartesian[i, 2] :
                    nodes_well4.append((i, r))
                if NL_Cartesian[i, 2] < 20:
                    nodes_well44b.append((i, r))

        return nodes_well1, nodes_well2, nodes_well3, nodes_well4, nodes_well22b, nodes_well33b, nodes_well44b


    def wellsprincipalsttress(self, principal_NODES, vector_NODES, nodes_well1, nodes_well2, nodes_well3, nodes_well4, nodes_well22b, nodes_well33b, nodes_well44b):

        stress_matrix_w1 = np.zeros([1, 2])
        stress_matrix_w2 = np.zeros([1, 2])
        stress_matrix_w3 = np.zeros([1, 2])
        stress_matrix_w4 = np.zeros([1, 2])

        stress_matrix_w22b = np.zeros([1, 2])
        stress_matrix_w33b = np.zeros([1, 2])
        stress_matrix_w44b = np.zeros([1, 2])
        stress_total_magnitude = np.zeros([7, 2])

        stress_direction_w1 = np.zeros([2, 2])
        stress_direction_w2 = np.zeros([2, 2])
        stress_direction_w3 = np.zeros([2, 2])
        stress_direction_w4 = np.zeros([2, 2])

        stress_direction_w22b = np.zeros([2, 2])
        stress_direction_w33b = np.zeros([2, 2])
        stress_direction_w44b = np.zeros([2, 2])

        stress_matrix_w1[:, 0] = (principal_NODES[nodes_well1[0][0], 0] + principal_NODES[nodes_well1[1][0], 0])/2
        stress_matrix_w1[:, 1] = (principal_NODES[nodes_well1[0][0], 1] + principal_NODES[nodes_well1[1][0], 1])/2

        stress_matrix_w2[:, 0] = principal_NODES[nodes_well2[0][0], 0]
        stress_matrix_w2[:, 1] = principal_NODES[nodes_well2[0][0], 1]

        stress_matrix_w3[:, 0] = (principal_NODES[nodes_well3[0][0], 0] + principal_NODES[nodes_well3[1][0], 0])/2
        stress_matrix_w3[:, 1] = (principal_NODES[nodes_well3[0][0], 1] + principal_NODES[nodes_well3[1][0], 1])/2

        stress_matrix_w4[:, 0] = (principal_NODES[nodes_well4[0][0], 0])                       #+ principal_NODES[nodes_well4[1][0], 0])/2
        stress_matrix_w4[:, 1] = (principal_NODES[nodes_well4[0][0], 1])                      #+ principal_NODES[nodes_well4[1][0], 1])/2

        stress_matrix_w22b[:, 0] = principal_NODES[nodes_well22b[0][0], 0]
        stress_matrix_w22b[:, 1] = principal_NODES[nodes_well22b[0][0], 1]

        stress_matrix_w33b[:, 0] = (principal_NODES[nodes_well33b[0][0], 0] + principal_NODES[nodes_well33b[1][0], 0])/2
        stress_matrix_w33b[:, 1] = (principal_NODES[nodes_well33b[0][0], 1] + principal_NODES[nodes_well33b[1][0], 1])/2

        stress_matrix_w44b[:, 0] = (principal_NODES[nodes_well44b[0][0], 0])               #+ principal_NODES[nodes_well44b[1][0], 0])/2
        stress_matrix_w44b[:, 1] = (principal_NODES[nodes_well44b[0][0], 1])              #+ principal_NODES[nodes_well44b[1][0], 1])/2

        stress_total_magnitude[0, :] = stress_matrix_w1
        stress_total_magnitude[1, :] = stress_matrix_w2
        stress_total_magnitude[2, :] = stress_matrix_w3
        stress_total_magnitude[3, :] = stress_matrix_w4
        stress_total_magnitude[4, :] = stress_matrix_w22b
        stress_total_magnitude[5, :] = stress_matrix_w33b
        stress_total_magnitude[6, :] = stress_matrix_w44b


        stress_direction_w1[:, :] = (vector_NODES[nodes_well1[0][0], 0, :, :]+ vector_NODES[nodes_well1[1][0], 0, :, :])/2
        stress_direction_w2[:, :] = vector_NODES[nodes_well2[0][0], 0, :, :]
        stress_direction_w3[:, :] = (vector_NODES[nodes_well3[0][0], 0, :, :] + vector_NODES[nodes_well3[1][0], 0, :, :])/2
        stress_direction_w4[:, :] = vector_NODES[nodes_well4[0][0], 0, :, :]
        stress_direction_w22b[:, :] = vector_NODES[nodes_well22b[0][0], 0, :, :]
        stress_direction_w33b[:, :] = (vector_NODES[nodes_well33b[0][0], 0, :, :] + vector_NODES[nodes_well33b[1][0], 0, :, :])/2
        stress_direction_w44b[:, :] = vector_NODES[nodes_well44b[0][0], 0, :, :]


        return stress_matrix_w1, stress_matrix_w2, stress_matrix_w3, stress_matrix_w4, stress_matrix_w22b, stress_matrix_w33b, stress_matrix_w44b, stress_direction_w1, \
        stress_direction_w2, stress_direction_w3, stress_direction_w4, stress_direction_w22b, stress_direction_w33b, stress_direction_w44b, stress_total_magnitude


    def wellsdata_stress(self):

        ############below_red Bulli seam ##############

        #vect_w1u_input = non
        vect_w22b_input = np.array([[-math.sin(math.pi/9), -math.cos(math.pi/9)],               # angle is 20 degrees, ED17
                                    [math.cos(math.pi/9), -math.sin(math.pi/9)]])

        vect_w33b_input = np.array([[math.sin(0), -math.cos(0)],                                          # angle is 180 or 0 degrees ED18
                                    [math.cos(0), -math.sin(0)]])

        vect_w44b_input = np.array([[-math.sin(math.pi/9), -math.cos(math.pi/9)],                         # angle is 20 degrees, ED35
                                    [math.cos(math.pi/9), -math.sin(math.pi/9)]])


        # ############above Bulli seam ##############


        vect_w1_input = np.array([[math.sin(math.pi/9), -math.cos(math.pi/9)],               # angle is 20 degrees, EAW6
                                  [math.cos(math.pi/9), math.sin(math.pi/9)]])

        vect_w2_input = np.array([[-math.sin(math.pi/4), -math.cos(math.pi/4)],               # angle is 45 degrees, ED17
                                  [math.cos(math.pi/4), -math.sin(math.pi/4)]])

        vect_w3_input = np.array([[-math.sin(math.pi/6), -math.cos(math.pi/6)],                       #angle is 60 degrees, ED18
                                [math.cos(math.pi/6), -math.sin(math.pi/6)]])

        vect_w4_input = np.array([[math.sin(math.pi/6), -math.cos(math.pi/6)],                         # angle is 30 degrees, ED35
                                [math.cos(math.pi/6), math.sin(math.pi/6)]])


        w2b_exp = vect_w22b_input[1, 1]/vect_w22b_input[0, 1]
        w3b_exp = vect_w33b_input[1, 1]/vect_w33b_input[0, 1]
        w4b_exp = vect_w44b_input[1, 1]/vect_w44b_input[0, 1]
        w1a_exp = vect_w1_input[1, 1]/vect_w1_input[0, 1]
        w2a_exp = vect_w2_input[1, 1]/vect_w2_input[0, 1]
        w3a_exp = vect_w3_input[1, 1]/vect_w3_input[0, 1]
        w4a_exp = vect_w4_input[1, 1]/vect_w4_input[0, 1]

        atan_2b = math.atan(w2b_exp)
        atan_3b = math.atan(w3b_exp)
        atan_4b = math.atan(w4b_exp)
        atan_1a = math.atan(w1a_exp)
        atan_2a = math.atan(w2a_exp)
        atan_3a = math.atan(w3a_exp)
        atan_4a = math.atan(w4a_exp)


        return atan_2b, atan_3b, atan_4b, atan_1a, atan_2a, atan_3a, atan_4a





