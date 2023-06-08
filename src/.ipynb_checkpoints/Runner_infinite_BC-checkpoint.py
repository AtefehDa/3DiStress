import math
import sys
import timeit
from turtle import shape

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d

from scipy import linalg, optimize
import pyvista as pv
from src.test import *

startTime = timeit.default_timer()
pd.set_option('precision', 4)
# startTime = datetime.now()
np.set_printoptions(precision=4, suppress=True)
rw = 0.1
re = 3.08
p = 9
m = 12
k = 2
PD = 3
# GPE = 4
E = 5
nu = 0.3
s = 0
sigma_H = 0.016
sigma_h = 0.014
pwell = 0.012
NoN = (p + 1) * (m + 1) * (k + 1)
NoE = p * m * k
# x_el_infinite = np.array([[6.87,])8.9617],
#                 [8.244, 46.754],
#                 [0, 47.4753],
#                 [0, 39.5627]])
np.set_printoptions(threshold=sys.maxsize)

object_mesh1 = MESH.Geometry(rw, re, p, m, k, PD, NoN, NoE)
NL, EL = object_mesh1.drawmesh()
EL = EL.astype(int)
NoE = np.size(EL, 0)
NPE = np.size(EL, 1)
x_coordinate = NL[:, 0]*np.cos(NL[:, 1])
y_coordinate = NL[:, 0]*np.sin(NL[:, 1])
z_coordinate = NL[:, 2]

NL_Cartesian = np.zeros([np.size(NL, 0), PD])
NL_Cartesian[:, 0] = x_coordinate
NL_Cartesian[:, 1] = y_coordinate
NL_Cartesian[:, 2] = z_coordinate
# y_coordinate.reshape(np.size(NL_Cartesian, 0), 1)
print(EL)
print(np.size(z_coordinate, 0))
# object_stiff1 = ELS.Elemen_stiffnesscalculation(x_el_infinite, GPE, NPE, PD, E, nu)
# kelement = object_stiff1.elementstiffness()
# # df_kelement = pd.DataFrame(kelement)
# print(kelement)
# object_BC1 = BC_finite.Boundaryconditionsassignment(PD, NoN, m, p, s, sigma_H, sigma_h, pwell)
# ENL, DOFS, DOCS, nodalforce_final, nodalforce_final_well = object_BC1.BC(NL_Cartesian)
# object_GLstiff1 = GLOBS.globalstiffnesscalculation(PD, NPE, NoE, NoN, GPE, E, nu)
# K_global = object_GLstiff1.globalstiffness_matrix(ENL, EL, NL_Cartesian)
# # df_K_global = pd.DataFrame(K_global)
# # # print(df_K_global)
# object_dispandforce1 = FDISP.disp_force_assemble(NoN, PD)
# Up = object_dispandforce1.assemble_displacement(ENL)
# Fp = object_dispandforce1.assemble_forces(ENL)

# k_reduced = K_global[0:DOFS, 0:DOFS]
# k_up = K_global[0:DOFS, DOFS:DOCS+DOFS]
# k_pu = K_global[DOFS:DOCS+DOFS, 0:DOFS]
# k_pp = K_global[DOFS:DOCS+DOFS, DOFS:DOCS+DOFS]

# force = Fp - (k_up @ Up)
# # dett = np.linalg.det(k_reduced)
# Uu_d = scipy.sparse.linalg.cg(k_reduced, force)
# Uu = Uu_d[0]
# # innnv = np.linalg.inv(k_reduced)
# # Uu = innnv @ force
# Uu = Uu.reshape(DOFS, 1)
# print(Uu)
# Fu = (k_pu @ Uu) + (k_pp @ Up)

# ENL = object_dispandforce1.updatenodes(ENL, Uu, Fu)
# object_postprocess1 = gradientofU.DoingPostprocess(PD, NoN, NoE, NPE, GPE, E, nu)
# disp, uxx_final, uxy_final, uyx_final, uyy_final  = object_postprocess1.element_postprocess(NL_Cartesian, EL, ENL)
# dispx, dispy, xnew, ynew, strain_xx, strain_yy, strain_xy, strain_yx, stress_xx, stress_xy, stress_yx, stress_yy = object_postprocess1.postprocess(NL_Cartesian, EL, ENL)
# print(f"ENL is \n {ENL}")

# STRESS_NEW_X = np.zeros([NoE, NPE])
# for i in range(0, NoE):
#     STRESS_NEW_X[i, :] = stress_xx[EL[i, :]-1]
# stress_xxnew = STRESS_NEW_X.T

# disp_alongx = []
# stress_rr_alongx = []
# NL_CARTESIn = []
# sigma_yy_n = []
# for i in range(1, NoN, 10):
#     stress_rr_alongx.append(stress_xx[i-1])
#     NL_CARTESIn.append(NL_Cartesian[i-1, 0])
#     disp_alongx.append(ENL[i-1, 8])
#     sigma_yy_n.append(stress_yy[i-1])

# stress_rr_alongx = np.array(stress_])_alongx)
# NL_CARTESIn = np.array(NL_CART])In)
# disp_alongx = np.array(disp_al])gx)
# sigma_yy_n = np.array(sigma_y])n)

# stress_p_2 = []
# r_axis_p_2 = []
# disp_r_p_2 = []
# disp_r_teta_p_6 = []
# stress_r_teta_p_6 = []
# r_axis_teta_p_6 = []
# teta_r_516 = []
# sigma_xy = []

# for i in range(1, NoN+1):
#     if NL[i-1, 1] == math.pi/6:
#         r_axis_teta_p_6.append(NL[i-1, 0])
#         disp_r_teta_p_6.append(ENL[i-1, 8])
#         stress_r_teta_p_6.append(stress_xx[i-1])

#     if NL[i-1, 1] == math.pi/2:
#         disp_r_p_2.append(ENL[i-1, 9])
#         r_axis_p_2.append(NL_Cartesian[i-1, 1])
#         stress_p_2.append(stress_yy[i-1])

#     if 0.515 < NL[i-1, 0] < 0.517:
#         teta_r_516.append(NL[i-1, 1])
#         sigma_xy.append(stress_xy[i-1])


# r_axis_teta_p_6 = np.array(r_axis_])ta_p_6)
# disp_r_teta_p_6 = np.array(disp_r_])ta_p_6)
# stress_r_teta_p_6 = np.array(stress_])teta_p_6)
# r_axis_p_2 = np.array(r_axis_])2)
# disp_r_p_2 = np.array(disp_r_])2)
# stress_p_2 = np.array(stress_])2)
# teta_r_516 = np.array(teta_r_])6)
# sigma_xy = np.array(sigma_x])
# print(teta_r_516)

# ############################################################################################################################################################################
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
fig1 = plt.figure()
ax = fig1.add_subplot(111, projection="3d")
# for I in range , color='blue', capacity=0.2(1, NoN + 1):
#     ax.annotate(count, xy = I-1], y[I-1]), size=
#     count += 1

#ount_element = 1
# fplt.annotate(count_element, xy = (EL[J, 0]-1] +EL[J, 11] +EL[J, 21] +EL[J,1])/4 ,
#           L[J, 2]-1] + y[EL[J, 3]-1])/4),
#                  c = 'blue', size=5)
#      count_element += 1

x0 , y0, zt0 = x_coordinate[EL[:, 0]-1], y_coordinate[EL[:, 0]-1], z_coordinate[EL[:, 0]-1]
x1 , y1, zt1 = x_coordinate[EL[:, 1]-1], y_coordinate[EL[:, 1]-1], z_coordinate[EL[:, 1]-1]
x2 , y2, zt2 = x_coordinate[EL[:, 2]-1], y_coordinate[EL[:, 2]-1], z_coordinate[EL[:, 2]-1]
x3 , y3, zt3 = x_coordinate[EL[:, 3]-1], y_coordinate[EL[:, 3]-1], z_coordinate[EL[:, 3]-1]
x4 , y4, zt4 = x_coordinate[EL[:, 4]-1], y_coordinate[EL[:, 4]-1], z_coordinate[EL[:, 4]-1]
x5 , y5, zt5 = x_coordinate[EL[:, 5]-1], y_coordinate[EL[:, 5]-1], z_coordinate[EL[:, 5]-1]
x6 , y6, zt6 = x_coordinate[EL[:, 6]-1], y_coordinate[EL[:, 6]-1], z_coordinate[EL[:, 6]-1]
x7 , y7, zt7 = x_coordinate[EL[:, 7]-1], y_coordinate[EL[:, 7]-1], z_coordinate[EL[:, 7]-1]

ax.plot(np.array([x0, x1]), np.array([y0, y1]))          #np.array([zt1, zt2])') 
# ax.plot(np.array([x0, x1]), np.array([y0, y1]))         #np.array([y0, y1])')        #np.array([zt0, zt1]))
# ax.plot(np.array([x1, x2]), np.array([y1, y2])) #np.array([zt1, zt2]))
# ax.plot(np.array([x2, x3]), np.array([y2, y3])) #np.array([zt2, zt3]))
# ax.plot(np.array([x3, x0]), np.array([y3, y0])) #np.array([zt3, zt0]))

# ax.plot(np.array([x4, x5]), np.array([y4, y5]), np.array([zt4, zt5]))
# ax.plot(np.array([x5, x6]), np.array([y5, y6]), np.array([zt5, zt6]))
# ax.plot(np.array([x6, x7]), np.array([y6, y7]), np.array([zt6, zt7]))
# ax.plot(np.array([x7, x4]), np.array([y7, y4]), np.array([zt7, zt4]))

# ax.plot(np.array([x0, x1]), np.array([y0, y1]), np.array([zt0, zt1]))
# ax.plot(np.array([x1, x5]), np.array([y1, y5]), np.array([zt1, zt5]))
# ax.plot(np.array([x5, x4]), np.array([y5, y4]), np.array([zt5, zt4]))
# ax.plot(np.array([x4, x0]), np.array([y4, y0]), np.array([zt4, zt0]))

# ax.plot(np.array([x3, x2]), np.array([y3, y2]), np.array([zt3, zt2]))
# ax.plot(np.array([x2, x6]), np.array([y2, y6]), np.array([zt2, zt6]))
# ax.plot(np.array([x6, x7]), np.array([y6, y7]), np.array([zt6, zt7]))
# ax.plot(np.array([x7, x3]), np.array([y7, y3]), np.array([zt7, zt3]))

# ax.plot(np.array([x2, x1]), np.array([y2, y1]), np.array([zt2, zt1]))
# ax.plot(np.array([x1, x5]), np.array([y1, y5]), np.array([zt1, zt5]))
# ax.plot(np.array([x5, x6]), np.array([y5, y6]), np.array([zt5, zt6]))
# ax.plot(np.array([x6, x2]), np.array([y6, y2]), np.array([zt6, zt2]))

# ax.plot(np.array([x3, x0]), np.array([y3, y0]), np.array([zt3, zt0]))
# ax.plot(np.array([x0, x4]), np.array([y0, y4]), np.array([zt0, zt4]))
# ax.plot(np.array([x4, x7]), np.array([y4, y7]), np.array([zt4, zt7]))
# ax.plot(np.array([x7, x3]), np.array([y7, y3]), np.array([zt7, zt3]))
# # #
# plt.show()
plt.savefig('src/figures/mesh3d.png')
# print(NL_Cartesian)
# print(x_coordinate)
# print(z_coordinate)
# ax.plot(NL_Cartesian[390:)800], y=NL_Cartesia)[3:780, 1], zt1=NL_Ca)te', opacity=0.20)])
# ax.show()
# print(NL_Cartesian[260:,2])
# # fig1 = go.Figure(NL_Cartesian[0:130,0 y=NL_Cartesian[0:131], zt1=NL_Cartesian[0y=0.20)])
# # fig2 = go.Figure(NL_Cartesian[130:260], y=NL_Cartesian[13260,1], zt1=NL_Cartesiopacity=0.20)])
# # fig3 = go.Figure(NL_Cartesian[260:,0]y=NL_Cartesian[260:,, zt1=NL_Cartesian[260.20)])
# # # ax.plot().set_subplots(fi)1, fig2)
# # fig.sh)w()

# ax = plt.axes()rojection='3d')

# ax.plot(NL_Cartesian[:,0], NL_Cartesian[:,1], NL_Cartesian[:,2], color='black')
# plt.show()
# plt.savefig('src/figures/mesh.png')

# dd = EL[:,:4]
# gg= EL[108:,4:8]

# print(EL)
# print(dd)
# dddd = dd
# print(dddd)
# print(gg)
# ax = plt.axes(projection='3d')
# gg = gg
# ax.plot(NL_Cartesian[dddd):]-1, 0], NL_Cartesian[dddd[:]-1, 1], NL_)[dddd[:]-1, 2], color='black')
# ax.plot(NL_Cartesian[gg[:)-1, 0], NL_Cartesia)[gg[:]-1, 1], NL_Cart)sian[gg[:]-1, 2], color='black')
# plt.show()
# plt.savefig('src/figures/mesh333.png')


# # # print(EL[108, :])
# fig2 = plt.figure()
# ax1 = plt.axes(projection='3d')
# ax1.plot3D(NL_Cartesian[EL[:]-1, 0], NL_Cartesian[EL[:]-1, 1], NL_Cartesian[EL[:]-1, 2], color='black')

# plt.show()
# plt.savefig('src/figures/mesh333.png')