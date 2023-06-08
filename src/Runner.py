import math
import sys
import timeit

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypardiso
import scipy
import scipy.sparse as sp
from scipy import linalg, optimize, sparse

from src.test import *

startTime = timeit.default_timer()
np.set_printoptions(precision=4, suppress=True)
err_stressmatch = np.zeros([5*5, 5])
p = 69
m = 75
k = 4
PD = 3
GPE = 8

E_el = np.loadtxt("src/data/E_ELEM.txt", dtype=float)
nu_elm = np.loadtxt("src/data/NU_ELEM.txt", dtype=float)
E_elm = E_el*1e9

E_nodeB = np.loadtxt("src/data/ENODE.txt", dtype=float)
nu_node = np.loadtxt("src/data/NUNODE.txt", dtype=float)
E_node = E_nodeB*1e9

object_mesh1 = MESH.Geometry(PD)
NL, EL = object_mesh1.drawmesh()
EL = EL.astype(int)
NPE = np.size(EL, 1)
NoE = np.size(EL, 0)
NL_Cartesian = NL
NoN = np.size(NL_Cartesian,0)

sigmav = 27*1e6
max_y = max(NL_Cartesian[:, 1])
min_x = min(NL_Cartesian[:, 0])
min_y = min(NL_Cartesian[:, 1])
max_x = max(NL_Cartesian[:, 0])
stress_magnitude = np.zeros([5*5*7, 4])
inum = 0
for iwell in range(0, 1):
    for bb in range(0, 1):
        inum+=1
        dispp =  0.012 + iwell*0.002
        disssp = dispp/2 + bb*dispp/4
        object_BC1 = BC_gravity.Boundaryconditionsassignment(PD, NoN, NPE, NoE, EL, m, p, k, sigmav, dispp, disssp, max_y, min_x, min_y, max_x, E_node)
        ENL, DOFS, DOCS = object_BC1.BC(NL_Cartesian)
        object_GLstiff1 = GLOBS.globalstiffnesscalculation(PD, NPE, NoE, NoN, GPE, E_elm, nu_elm)
        K_global = object_GLstiff1.globalstiffness_matrix(ENL, EL, NL_Cartesian)
        object_dispandforce1 = FDISP.disp_force_assemble(NoN, PD)
        Up = object_dispandforce1.assemble_displacement(ENL)
        Fp = object_dispandforce1.assemble_forces(ENL)
        a = K_global[0:DOFS, 0:DOFS]
        k_up = K_global[0:DOFS, DOFS:DOCS+DOFS]
        k_pu = K_global[DOFS:DOCS+DOFS, 0:DOFS]
        k_pp = K_global[DOFS:DOCS+DOFS, DOFS:DOCS+DOFS]
        force = Fp - (k_up @ Up)
        np.savetxt(f"src/results/ENL{inum-1}.csv", ENL, delimiter=',')

        sA = sparse.csr_matrix(a)
        Uu = pypardiso.spsolve(sA, force)
        Uu = Uu.reshape(DOFS, 1)
        Fu = (k_pu @ Uu) + (k_pp @ Up)
        ENL = object_dispandforce1.updatenodes(ENL, Uu, Fu)
        object_patch = patch.DoingPostprocess(PD, NoN, NoE, NPE, GPE, E_node, nu_node)
        uxx_final, uyx_final, uzx_final, uxy_final, uyy_final, uzy_final, uxz_final, uyz_final, uzz_final = object_patch.element_postprocess(NL_Cartesian, EL, ENL)
        strain_xx, strain_yx, strain_zx, strain_xy, strain_yy, strain_zy, \
        strain_xz, strain_yz, strain_zz, stress_xx, stress_yx, stress_zx, stress_xy, stress_yy, \
        stress_zy, stress_xz, stress_yz, stress_zz = object_patch.postprocess(NL_Cartesian, EL, ENL)
        sTRESS_new = np.zeros([NoN, 1, 2, 2])
        sTRESS_new[0:NoN, 0, 0, 0] = stress_xx[:, 0]
        sTRESS_new[0:NoN, 0, 0, 1] = stress_xy[:, 0]
        sTRESS_new[0:NoN, 0, 1, 0] = stress_yx[:, 0]
        sTRESS_new[0:NoN, 0, 1, 1] = stress_yy[:, 0]

        principal_NODES = np.zeros([NoN, 2])
        vector_NODES = np.zeros([NoN, 1, 2, 2])
        for i in range(0, NoN):
            e1, v1 = np.linalg.eigh(sTRESS_new[i, 0, :, :])
            e1 = e1.reshape(1, 2)
            principal_NODES[i, :] = e1
            vector_NODES[i, 0, :, :] = v1[:, :]
        np.savetxt(f"src/results/principal_NODES{inum-1}.txt", principal_NODES)
        object_location1 = principalwell.principal(PD, NoN, NoE, NPE, GPE)
        nodes_well1, nodes_well2, nodes_well3, nodes_well4, nodes_well22b, nodes_well33b, nodes_well44b = object_location1.location(NL_Cartesian, EL)

        stress_matrix_w1, stress_matrix_w2, stress_matrix_w3, stress_matrix_w4, stress_matrix_w22b, stress_matrix_w33b, stress_matrix_w44b, stress_direction_w1, \
        stress_direction_w2, stress_direction_w3, stress_direction_w4, stress_direction_w22b, stress_direction_w33b, stress_direction_w44b, stress_total_magnitude\
        = object_location1.wellsprincipalsttress(principal_NODES, vector_NODES, nodes_well1, nodes_well2,\
        nodes_well3, nodes_well4, nodes_well22b, nodes_well33b, nodes_well44b)

        stress_magnitude[7*(inum-1):7*(inum-1)+7, 0] = iwell
        stress_magnitude[7*(inum-1):7*(inum-1)+7, 1] = bb
        stress_magnitude[7*(inum-1):7*(inum-1)+7, 2:] = stress_total_magnitude

        atan_2b, atan_3b, atan_4b, atan_1a, atan_2a, atan_3a, atan_4a = object_location1.wellsdata_stress()

        w2b_cal = math.atan(stress_direction_w22b[1, 1]/stress_direction_w22b[0, 1])
        w3b_cal = math.atan(stress_direction_w33b[1, 1]/stress_direction_w33b[0, 1])
        w4b_cal = math.atan(stress_direction_w44b[1, 1]/stress_direction_w44b[0, 1])
        w1a_cal = math.atan(stress_direction_w1[1, 1]/stress_direction_w1[0, 1])
        w2a_cal = math.atan(stress_direction_w2[1, 1]/stress_direction_w2[0, 1])
        w3a_cal = math.atan(stress_direction_w3[1, 1]/stress_direction_w3[0, 1])
        w4a_cal = math.atan(stress_direction_w4[1, 1]/stress_direction_w4[0, 1])

        well2b = (w2b_cal - atan_2b)
        well3b = (w3b_cal - atan_3b)
        well4b = (w4b_cal - atan_4b)
        well1a = (w1a_cal - atan_1a)
        well2a = (w2a_cal - atan_2a)
        well3a = (w3a_cal - atan_3a)
        well4a = (w4a_cal - atan_4a)
        MAE = (abs(well2b) + abs(well3b) + abs(well4b) + abs(well1a) + abs(well2a) + abs(well3a) + abs(well4a))/7    #MAE Error
        mean_data = (well2b+well3b+well4b+well1a+well2a+well3a+well4a)/7
        SD = math.sqrt(((well2b-mean_data)**2+(well3b-mean_data)**2+(well4b-mean_data)**2+(well1a-mean_data)**2+(well2a-mean_data)**2+(well3a-mean_data)**2+(well4a-mean_data)**2)/6)

        err_stressmatch[inum-1, 0] = iwell
        err_stressmatch[inum-1, 1] = bb
        err_stressmatch[inum-1, 2] = MAE
        err_stressmatch[inum-1, 3] = mean_data
        err_stressmatch[inum-1, 4] = SD

        fig, ax = plt.subplots()

        origin = [419.87, 1240.94]
        eig_vec1a = stress_direction_w1[:,0]
        eig_vec2a = stress_direction_w1[:,1]
        plt.quiver(*origin, *eig_vec2a, color=['k'], scale=10)   #minimum
        plt.text(219, 1040, 'well_No1')

        origin = [1001.67, 865.13]
        eig_vec22a = stress_direction_w2[:,0]
        eig_vec22aa = stress_direction_w2[:,1]
        plt.quiver(*origin, *eig_vec22aa, color=['k'], scale=10)   #minimum
        plt.text(800, 665, 'well_No2')

        origin = [288.83, 368.27]
        eig_vec33a = stress_direction_w3[:,0]
        eig_vec33aa = stress_direction_w3[:,1]
        plt.quiver(*origin, *eig_vec33aa, color=['k'], scale=10)
        plt.text(88, 168, 'well_No3')

        origin = [920.45, 270.6]
        eig_vec44a = stress_direction_w4[:,0]
        eig_vec44aa = stress_direction_w4[:,1]
        plt.quiver(*origin, *eig_vec44aa, color=['k'], scale=10)
        plt.text(720, 70, 'well_No4')
        # # # ####################below

        origin = [1001.67, 865.13]
        eig_vec222b = stress_direction_w22b[:,0]
        eig_vec222bb = stress_direction_w22b[:,1]
        plt.quiver(*origin, *eig_vec222bb, color=['r'], scale=10)   #minimum

        origin = [288.83, 368.27]
        eig_vec333b = stress_direction_w33b[:,0]
        eig_vec333bb = stress_direction_w33b[:,1]
        plt.quiver(*origin, *eig_vec333bb, color=['r'], scale=10)

        origin = [920.45, 270.6]
        eig_vec444b = stress_direction_w44b[:,0]
        eig_vec444bb = stress_direction_w44b[:,1]
        plt.quiver(*origin, *eig_vec444bb, color=['r'], scale=10)

        ax.set(xlim=(-100, 1600), ylim=(-100, 1600))
        plt.show()
        plt.savefig(f"src/results/stressdirections{inum-1}.png")


np.savetxt('src/results/stress_magnitude.csv', stress_magnitude, delimiter=',')
np.savetxt('src/results/err_stressmatch.csv', err_stressmatch, delimiter=',')



endTime = timeit.default_timer()
execution_time = endTime - startTime
print(f"this program took \n {execution_time} seconds ")

