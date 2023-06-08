
import math
import numpy as np
import pandas as pd

class DoingPostprocess:

    def __init__(self, PD, NoN, NoE, NPE, GPE, E_node, nu_node):
        self.PD = PD
        self.NoN = NoN
        self.NoE = NoE
        self.NPE = NPE
        self.GPE = GPE
        self.E_node = E_node
        self.nu_node =nu_node
    def postprocess(self, NL_Cartesian, EL, ENL):

        uxx_final, uyx_final, uzx_final, uxy_final, uyy_final, uzy_final, uxz_final, uyz_final, uzz_final \
        = DoingPostprocess.element_postprocess(self, NL_Cartesian, EL, ENL)

        strain_xx = uxx_final[:, 1]
        strain_yx = 1/2 * (uxy_final[:, 1] + uyx_final[:, 1])
        strain_zx = 1/2 * (uzx_final[:, 1] + uxz_final[:, 1])
        strain_xy = 1/2 * (uxy_final[:, 1] + uyx_final[:, 1])
        strain_yy = uyy_final[:, 1]
        strain_zy = 1/2 * (uzy_final[:, 1] + uyz_final[:, 1])
        strain_xz = 1/2 * (uzx_final[:, 1] + uxz_final[:, 1])
        strain_yz = 1/2 * (uzy_final[:, 1] + uyz_final[:, 1])
        strain_zz = uzz_final[:, 1]

        lambdaa = np.zeros([self.NoN, 1])
        Q = np.zeros([self.NoN, 1])
        DQ = np.zeros([self.NoN, 1])

        for i in range(0, self.NoN):
            lambdaa[i] = (self.E_node[i] * (1 - self.nu_node[i]))/((1 - 2 * self.nu_node[i]) * (1 + self.nu_node[i]))
            Q[i] = (self.E_node[i] * self.nu_node[i])/((1 - 2 * self.nu_node[i]) * (1 + self.nu_node[i]))
            DQ[i] = self.E_node[i]/(1 + self.nu_node[i])

        stress_xx = np.zeros([self.NoN, 1])
        stress_yx = stress_xy = np.zeros([self.NoN, 1])
        stress_yy = np.zeros([self.NoN, 1])
        stress_zy = stress_yz = np.zeros([self.NoN, 1])
        stress_zz = np.zeros([self.NoN, 1])
        stress_zx = stress_xz = np.zeros([self.NoN, 1])

        for i in range(0, self.NoN):

            stress_xx[i] = lambdaa[i] * strain_xx[i] + Q[i] * strain_yy[i] + Q[i] * strain_zz[i]
            stress_yx[i] = DQ[i] * strain_yx[i]
            stress_zx[i] = DQ[i] * strain_zx[i]
            stress_xy[i] = DQ[i] * strain_xy[i]
            stress_yy[i] = lambdaa[i] * strain_yy[i] + Q[i] * strain_xx[i] + Q[i] * strain_zz[i]
            stress_zy[i] = DQ[i] * strain_zy[i]
            stress_xz[i] = DQ[i] * strain_xz[i]
            stress_yz[i] = DQ[i] * strain_yz[i]
            stress_zz[i] = lambdaa[i] * strain_zz[i] + Q[i] * strain_xx[i] + Q[i] * strain_yy[i]


        return strain_xx, strain_yx, strain_zx, strain_xy, strain_yy, strain_zy, \
        strain_xz, strain_yz, strain_zz, stress_xx, stress_yx, stress_zx, stress_xy, stress_yy, stress_zy, stress_xz, stress_yz, stress_zz

    def element_postprocess(self, NL_Cartesian, EL, ENL):

        Num_of_patches = 0
        arr_patch = np.zeros([74*15096, 10])       #in 20 noded elements, you have 81 nodes in every patch, you want to obtain the disp gradient of 5 nodes in center separately
        arr_patch_center = np.zeros([7*15096, 10])

        for nodep in range(1, self.NoN+1):
            occur = np.count_nonzero(EL == nodep)
            if occur==8:
                patch = np.zeros([8, self.NPE], dtype=int)
                Num_of_patches += 1
                PA = np.argwhere(EL == nodep)
                patch[0:8, :] = EL[PA[0:8,0], :]

                A_patch = 0
                bsmall1 = 0
                bsmall2 = 0
                bsmall3 = 0
                bsmall4 = 0
                bsmall5 = 0
                bsmall6 = 0
                bsmall7 = 0
                bsmall8 = 0
                bsmall9 = 0

                for k in range(0, np.size(patch, 0)):
                    nlnew = patch[k, 0:self.NPE]
                    xn = np.zeros([self.NPE, self.PD])
                    u = np.zeros([self.NPE, self.PD])
                    xn[0:self.NPE, 0:self.PD] = NL_Cartesian[nlnew[0:self.NPE]-1, 0:self.PD]
                    for i in range(0, self.NPE):
                        for j in range(0, self.PD):
                            u[i, j] = ENL[nlnew[i]-1, 4*self.PD+j]                        #NODAL displacement values
                    trans_x = xn.T
                    for gp in range(1, self.GPE+1):
                        Jacobian = np.zeros([self.PD, self.PD])
                        grad_integral = np.zeros([self.PD, self.NPE])
                        (xi, eta, Si, alpha) = DoingPostprocess.Gausspoint(self, gp)
                        derivative = DoingPostprocess.grad_shapefunction(self, xi, eta, Si)
                        shape = DoingPostprocess.shape_functions(self, xi, eta, Si)
                        Jacobian = trans_x @ derivative.T
                        grad_integral = np.linalg.inv(Jacobian).T @ derivative
                        g =  grad_integral @ u                                  #gradient of u : each gauss point has a 2*2 gradient matrix:approximation of displacement
                        X_Y_Zk = shape.T @ xn                             ##approximation of coordinates of gauss point in physical domain
                        q = np.zeros([1,10])                                      #each gp has a g matrix which is the displacement gradient at gp
                        q[0, 0] = 1
                        q[0, 1] = X_Y_Zk[0, 0]
                        q[0, 2] = X_Y_Zk[0, 1]
                        q[0, 3] = X_Y_Zk[0, 2]
                        q[0, 4] = X_Y_Zk[0, 0] * X_Y_Zk[0, 1]
                        q[0, 5] = X_Y_Zk[0, 0] * X_Y_Zk[0, 2]
                        q[0, 6] = X_Y_Zk[0, 1] * X_Y_Zk[0, 2]
                        q[0, 7] = X_Y_Zk[0, 0] ** 2
                        q[0, 8] = X_Y_Zk[0, 1] ** 2
                        q[0, 9] = X_Y_Zk[0, 2] ** 2
                        q_trans = q.T
                        B1 = q_trans @ q
                        A_patch = A_patch + B1                                  #in each patch you earned the A_patch matrix,

                        bsmall1 = bsmall1 + q_trans * g[0, 0]
                        bsmall2 = bsmall2 + q_trans * g[0, 1]
                        bsmall3 = bsmall3 + q_trans * g[0, 2]
                        bsmall4 = bsmall4 + q_trans * g[1, 0]
                        bsmall5 = bsmall5 + q_trans * g[1, 1]
                        bsmall6 = bsmall6 + q_trans * g[1, 2]
                        bsmall7 = bsmall7 + q_trans * g[2, 0]
                        bsmall8 = bsmall8 + q_trans * g[2, 1]
                        bsmall9 = bsmall9 + q_trans * g[2, 2]

                asmall1 = np.linalg.inv(A_patch) @ bsmall1
                asmall2 = np.linalg.inv(A_patch) @ bsmall2
                asmall3 = np.linalg.inv(A_patch) @ bsmall3
                asmall4 = np.linalg.inv(A_patch) @ bsmall4
                asmall5 = np.linalg.inv(A_patch) @ bsmall5
                asmall6 = np.linalg.inv(A_patch) @ bsmall6
                asmall7 = np.linalg.inv(A_patch) @ bsmall7
                asmall8 = np.linalg.inv(A_patch) @ bsmall8
                asmall9 = np.linalg.inv(A_patch) @ bsmall9


                COOR_CENTER = NL_Cartesian[nodep-1, :]
                solution_center = np.array([1, COOR_CENTER[0], COOR_CENTER[1], COOR_CENTER[2], COOR_CENTER[0]*COOR_CENTER[1], \
                COOR_CENTER[0]*COOR_CENTER[2], COOR_CENTER[1]*COOR_CENTER[2], COOR_CENTER[0]**2, COOR_CENTER[1]**2, COOR_CENTER[2]**2])
                ucenter = np.zeros([1, (self.PD*self.PD) + 1])
                ucenter[0, 0] = nodep
                ucenter[0, 1] = solution_center @ asmall1
                ucenter[0, 2] = solution_center @ asmall2
                ucenter[0, 3] = solution_center @ asmall3
                ucenter[0, 4] = solution_center @ asmall4
                ucenter[0, 5] = solution_center @ asmall5
                ucenter[0, 6] = solution_center @ asmall6
                ucenter[0, 7] = solution_center @ asmall7
                ucenter[0, 8] = solution_center @ asmall8
                ucenter[0, 9] = solution_center @ asmall9
                arr_patch_center[(Num_of_patches-1)*7, :] = ucenter

                arrside = np.unique(patch.flatten())
                arrside = np.delete(arrside, np.where(arrside == nodep))
                coor_s = np.zeros([np.size(arrside), self.PD])
                coor_s = NL_Cartesian[arrside-1, :]

                patch_x1 = np.zeros([4, self.NPE])
                patch_x1 = patch[[1,3,5,7], :]

                patch_x2 = np.zeros([4, self.NPE])
                patch_x2 = patch[[0,2,4,6], :]

                patch_y1 = np.zeros([4, self.NPE])
                patch_y1 = patch[[0,1,4,5], :]

                patch_y2 = np.zeros([4, self.NPE])
                patch_y2 = patch[[2,3,6,7], :]
                uzpos = []
                uzneg = []
                u_xyp = []
                uside = np.zeros([74, 10])
                uside_patch = np.zeros([6, 10])
                for l in range(0, np.size(arrside)):
                    side_node = arrside[l]
                    if -0.001<NL_Cartesian[side_node-1, 0] - min(coor_s[:, 0])<0.001 or -0.001 < NL_Cartesian[side_node-1, 0] - max(coor_s[:, 0])<0.001 or\
                        -0.001< NL_Cartesian[side_node-1, 1] - min(coor_s[:, 1])<0.001 or -0.001<NL_Cartesian[side_node-1, 1] - max(coor_s[:, 1])<0.001 :
                        continue
                    occurssx1 = np.count_nonzero(patch_x1 == side_node)
                    occurssx2 = np.count_nonzero(patch_x2 == side_node)
                    occurssy1 = np.count_nonzero(patch_y1 == side_node)
                    occurssy2 = np.count_nonzero(patch_y2 == side_node)
                    if occurssx1 == 4 or occurssx2 == 4 or occurssy1 == 4 or occurssy2==4:
                        u_xyp.append(side_node)

                    if -0.001 < NL_Cartesian[side_node-1, 0] - NL_Cartesian[nodep-1, 0]< 0.001 and -0.001< NL_Cartesian[side_node-1, 1] - NL_Cartesian[nodep-1, 1]< 0.001:
                        dist = NL_Cartesian[nodep-1, 2] - NL_Cartesian[side_node-1, 2]
                        if dist>0:
                            uzpos.append((side_node, dist))
                        else:
                            uzneg.append((side_node, dist))


                uzpos = sorted(uzpos, key=lambda distant: distant[1])
                uzneg = sorted(uzneg, key=lambda distant: distant[1])
                u_xyp.append(uzpos[0][0])
                u_xyp.append(uzneg[1][0])

                for i in range(0, len(u_xyp)):
                    arrside = np.delete(arrside, np.where(arrside == u_xyp[i]))

                for l in range(0, len(u_xyp)):
                    side_center = u_xyp[l]
                    COOR_SIDE = NL_Cartesian[side_center-1, :]
                    solution_side_center = np.array([1, COOR_SIDE[0], COOR_SIDE[1], COOR_SIDE[2], COOR_SIDE[0]*COOR_SIDE[1], \
                    COOR_SIDE[0]*COOR_SIDE[2], COOR_SIDE[1]*COOR_SIDE[2], COOR_SIDE[0]**2, COOR_SIDE[1]**2, COOR_SIDE[2]**2])
                    uside_patch[l, 0] = side_center
                    uside_patch[l, 1] = solution_side_center  @ asmall1
                    uside_patch[l, 2] = solution_side_center  @ asmall2
                    uside_patch[l, 3] = solution_side_center  @ asmall3
                    uside_patch[l, 4] = solution_side_center  @ asmall4
                    uside_patch[l, 5] = solution_side_center  @ asmall5
                    uside_patch[l, 6] = solution_side_center  @ asmall6
                    uside_patch[l, 7] = solution_side_center  @ asmall7
                    uside_patch[l, 8] = solution_side_center  @ asmall8
                    uside_patch[l, 9] = solution_side_center  @ asmall9

                for l in range(0, np.size(arrside)):
                    side_node = arrside[l]
                    COOR_SIDE = NL_Cartesian[side_node-1, :]
                    solution_side_node = np.array([1, COOR_SIDE[0], COOR_SIDE[1], COOR_SIDE[2], COOR_SIDE[0]*COOR_SIDE[1], \
                    COOR_SIDE[0]*COOR_SIDE[2], COOR_SIDE[1]*COOR_SIDE[2], COOR_SIDE[0]**2, COOR_SIDE[1]**2, COOR_SIDE[2]**2])
                    uside[l, 0] = side_node
                    uside[l, 1] = solution_side_node  @ asmall1
                    uside[l, 2] = solution_side_node  @ asmall2
                    uside[l, 3] = solution_side_node  @ asmall3
                    uside[l, 4] = solution_side_node  @ asmall4
                    uside[l, 5] = solution_side_node  @ asmall5
                    uside[l, 6] = solution_side_node  @ asmall6
                    uside[l, 7] = solution_side_node  @ asmall7
                    uside[l, 8] = solution_side_node  @ asmall8
                    uside[l, 9] = solution_side_node  @ asmall9

                arr_patch_center[(Num_of_patches-1)*7+1:(Num_of_patches-1)*7+7, :] = uside_patch
                arr_patch[(Num_of_patches-1)*74:Num_of_patches*74, :] = uside

        for i in range(0, np.size(arr_patch_center, 0)):
            if arr_patch[i, 0]==arr_patch_center[i,0]:
                arr_patch[i,:]=0

        arr_patch = arr_patch[~np.all(arr_patch == 0, axis=1)]

        df1 = pd.DataFrame(arr_patch, columns = ["Nodes", "xx", "yx", "zx", "xy", "yy", "zy", "xz", "yz", "zz"])
        df2 = pd.DataFrame(arr_patch_center, columns = ["Nodes", "xx", "yx", "zx", "xy", "yy", "zy", "xz", "yz", "zz"])

        average_1 = df1.groupby('Nodes').mean().reset_index()
        average_2 = df2.groupby('Nodes').mean().reset_index()

        u_side = average_1.to_numpy()
        u_center = average_2.to_numpy()

        arr_final = np.concatenate((u_side, u_center), axis=0)

        uxx_final = arr_final[:, 0:2]
        uyx_final = arr_final[:, [0, 2]]
        uzx_final = arr_final[:, [0, 3]]
        uxy_final = arr_final[:, [0, 4]]
        uyy_final = arr_final[:, [0, 5]]
        uzy_final = arr_final[:, [0, 6]]
        uxz_final = arr_final[:, [0, 7]]
        uyz_final = arr_final[:, [0, 8]]
        uzz_final = arr_final[:, [0, 9]]


        return uxx_final, uyx_final, uzx_final, uxy_final, uyy_final, uzy_final, uxz_final, uyz_final, uzz_final

    def Gausspoint(self, gp):

        if gp == 1:
            xi = -1/math.sqrt(3)
            eta = -1/math.sqrt(3)
            Si = -1/math.sqrt(3)
            alpha = 1
        if gp == 2:
            xi = 1/math.sqrt(3)
            eta = -1/math.sqrt(3)
            Si = -1/math.sqrt(3)
            alpha = 1
        if gp == 3:
            xi = 1/math.sqrt(3)
            eta = 1/math.sqrt(3)
            Si = -1/math.sqrt(3)
            alpha = 1
        if gp == 4:
            xi = -1/math.sqrt(3)
            eta = 1/math.sqrt(3)
            Si = -1/math.sqrt(3)
            alpha = 1
        if gp == 5:
            xi = -1/math.sqrt(3)
            eta = -1/math.sqrt(3)
            Si = 1/math.sqrt(3)
            alpha = 1
        if gp == 6:
            xi = 1/math.sqrt(3)
            eta = -1/math.sqrt(3)
            Si = 1/math.sqrt(3)
            alpha = 1
        if gp == 7:
            xi = 1/math.sqrt(3)
            eta = 1/math.sqrt(3)
            Si = 1/math.sqrt(3)
            alpha = 1
        if gp == 8:
            xi = -1/math.sqrt(3)
            eta = 1/math.sqrt(3)
            Si = 1/math.sqrt(3)
            alpha = 1

        return (xi, eta, Si, alpha)

    def grad_shapefunction(self, xi, eta, Si):
        derivative = np.zeros([self.PD, self.NPE])
        derivative[0, 0] = -1/8*(1-eta)*(1-Si)*(-xi-eta-Si-2) + (-1/8*(1-xi)*(1-eta)*(1-Si))
        derivative[0, 1] = 1/8*(1-eta)*(1-Si)*(xi-eta-Si-2) + (1/8*(1+xi)*(1-eta)*(1-Si))
        derivative[0, 2] = 1/8*(1+eta)*(1-Si)*(xi+eta-Si-2) + (1/8*(1+xi)*(1+eta)*(1-Si))
        derivative[0, 3] = -1/8*(1+eta)*(1-Si)*(-xi+eta-Si-2) + (-1/8*(1-xi)*(1+eta)*(1-Si))
        derivative[0, 4] = -1/8*(1-eta)*(1+Si)*(-xi-eta+Si-2) + (-1/8*(1-xi)*(1-eta)*(1+Si))
        derivative[0, 5] = 1/8*(1-eta)*(1+Si)*(xi-eta+Si-2) + (1/8*(1+xi)*(1-eta)*(1+Si))
        derivative[0, 6] = 1/8*(1+eta)*(1+Si)*(xi+eta+Si-2) + (1/8*(1+xi)*(1+eta)*(1+Si))
        derivative[0, 7] = -1/8*(1+eta)*(1+Si)*(-xi+eta+Si-2) + (-1/8*(1-xi)*(1+eta)*(1+Si))
        derivative[0, 8] = -1/2*(xi)*(1-eta)*(1-Si)
        derivative[0, 9] = 1/4*(1-eta**2)*(1-Si)
        derivative[0, 10] = -1/2*(xi)*(1+eta)*(1-Si)
        derivative[0, 11] = -1/4*(1-eta**2)*(1-Si)
        derivative[0, 12] = -1/4*(1-eta)*(1-Si**2)
        derivative[0, 13] = 1/4*(1-eta)*(1-Si**2)
        derivative[0, 14] = 1/4*(1+eta)*(1-Si**2)
        derivative[0, 15] = -1/4*(1+eta)*(1-Si**2)
        derivative[0, 16] = -1/2*(xi)*(1-eta)*(1+Si)
        derivative[0, 17] = 1/4*(1-eta**2)*(1+Si)
        derivative[0, 18] = -1/2*(xi)*(1+eta)*(1+Si)
        derivative[0, 19] = -1/4*(1-eta**2)*(1+Si)

        derivative[1, 0] = -1/8*(1-xi)*(1-Si)*(-xi-eta-Si-2) + (-1/8*(1-xi)*(1-eta)*(1-Si))
        derivative[1, 1] = -1/8*(1+xi)*(1-Si)*(xi-eta-Si-2) + (-1/8*(1+xi)*(1-eta)*(1-Si))
        derivative[1, 2] = 1/8*(1+xi)*(1-Si)*(xi+eta-Si-2) + (1/8*(1+xi)*(1+eta)*(1-Si))
        derivative[1, 3] = 1/8*(1-xi)*(1-Si)*(-xi+eta-Si-2) + (1/8*(1-xi)*(1+eta)*(1-Si))
        derivative[1, 4] = -1/8*(1-xi)*(1+Si)*(-xi-eta+Si-2) + (-1/8*(1-xi)*(1-eta)*(1+Si))
        derivative[1, 5] = -1/8*(1+xi)*(1+Si)*(xi-eta+Si-2) + (-1/8*(1+xi)*(1-eta)*(1+Si))
        derivative[1, 6] = 1/8*(1+xi)*(1+Si)*(xi+eta+Si-2) + (1/8*(1+xi)*(1+eta)*(1+Si))
        derivative[1, 7] = 1/8*(1-xi)*(1+Si)*(-xi+eta+Si-2) + (1/8*(1-xi)*(1+eta)*(1+Si))
        derivative[1, 8] = -1/4*(1-xi**2)*(1-Si)
        derivative[1, 9] = -1/2*(eta)*(1+xi)*(1-Si)
        derivative[1, 10] = 1/4*(1-xi**2)*(1-Si)
        derivative[1, 11] = -1/2*(eta)*(1-xi)*(1-Si)
        derivative[1, 12] = -1/4*(1-xi)*(1-Si**2)
        derivative[1, 13] = -1/4*(1+xi)*(1-Si**2)
        derivative[1, 14] = 1/4*(1+xi)*(1-Si**2)
        derivative[1, 15] = 1/4*(1-xi)*(1-Si**2)
        derivative[1, 16] = -1/4*(1-xi**2)*(1+Si)
        derivative[1, 17] = -1/2*(eta)*(1+xi)*(1+Si)
        derivative[1, 18] = 1/4*(1-xi**2)*(1+Si)
        derivative[1, 19] = -1/2*(eta)*(1-xi)*(1+Si)

        derivative[2, 0] = -1/8*(1-xi)*(1-eta)*(-xi-eta-Si-2) + (-1/8*(1-xi)*(1-eta)*(1-Si))
        derivative[2, 1] = -1/8*(1+xi)*(1-eta)*(xi-eta-Si-2) + (-1/8*(1+xi)*(1-eta)*(1-Si))
        derivative[2, 2] = -1/8*(1+xi)*(1+eta)*(xi+eta-Si-2) + (-1/8*(1+xi)*(1+eta)*(1-Si))
        derivative[2, 3] = -1/8*(1-xi)*(1+eta)*(-xi+eta-Si-2) + (-1/8*(1-xi)*(1+eta)*(1-Si))
        derivative[2, 4] = 1/8*(1-xi)*(1-eta)*(-xi-eta+Si-2) + (1/8*(1-xi)*(1-eta)*(1+Si))
        derivative[2, 5] = 1/8*(1+xi)*(1-eta)*(xi-eta+Si-2) + (1/8*(1+xi)*(1-eta)*(1+Si))
        derivative[2, 6] = 1/8*(1+xi)*(1+eta)*(xi+eta+Si-2) + (1/8*(1+xi)*(1+eta)*(1+Si))
        derivative[2, 7] = 1/8*(1-xi)*(1+eta)*(-xi+eta+Si-2) + (1/8*(1-xi)*(1+eta)*(1+Si))
        derivative[2, 8] = -1/4*(1-xi**2)*(1-eta)
        derivative[2, 9] = -1/4*(1-eta**2)*(1+xi)
        derivative[2, 10] = -1/4*(1-xi**2)*(1+eta)
        derivative[2, 11] = -1/4*(1-eta**2)*(1-xi)
        derivative[2, 12] = -1/2*(Si)*(1-xi)*(1-eta)
        derivative[2, 13] = -1/2*(Si)*(1+xi)*(1-eta)
        derivative[2, 14] = -1/2*(Si)*(1+xi)*(1+eta)
        derivative[2, 15] = -1/2*(Si)*(1-xi)*(1+eta)
        derivative[2, 16] = 1/4*(1-xi**2)*(1-eta)
        derivative[2, 17] = 1/4*(1-eta**2)*(1+xi)
        derivative[2, 18] =  1/4*(1-xi**2)*(1+eta)
        derivative[2, 19] = 1/4*(1-eta**2)*(1-xi)

        return derivative

    def shape_functions(self, xi, eta, Si):
        shape = np.zeros([self.NPE, 1])

        shape[0, 0] = 1/8*(1-xi)*(1-eta)*(1-Si)*(-xi-eta-Si-2)
        shape[1, 0] = 1/8*(1+xi)*(1-eta)*(1-Si)*(xi-eta-Si-2)
        shape[2, 0] = 1/8*(1+xi)*(1+eta)*(1-Si)*(xi+eta-Si-2)
        shape[3, 0] = 1/8*(1-xi)*(1+eta)*(1-Si)*(-xi+eta-Si-2)
        shape[4, 0] = 1/8*(1-xi)*(1-eta)*(1+Si)*(-xi-eta+Si-2)
        shape[5, 0] = 1/8*(1+xi)*(1-eta)*(1+Si)*(xi-eta+Si-2)
        shape[6, 0] = 1/8*(1+xi)*(1+eta)*(1+Si)*(xi+eta+Si-2)
        shape[7, 0] = 1/8*(1-xi)*(1+eta)*(1+Si)*(-xi+eta+Si-2)
        shape[8, 0] = 1/4*(1-xi**2)*(1-eta)*(1-Si)
        shape[9, 0] = 1/4*(1+xi)*(1-eta**2)*(1-Si)
        shape[10, 0] = 1/4*(1-xi**2)*(1+eta)*(1-Si)
        shape[11, 0] = 1/4*(1-xi)*(1-eta**2)*(1-Si)
        shape[12, 0] = 1/4*(1-xi)*(1-eta)*(1-Si**2)
        shape[13, 0] = 1/4*(1+xi)*(1-eta)*(1-Si**2)
        shape[14, 0] = 1/4*(1+xi)*(1+eta)*(1-Si**2)
        shape[15, 0] = 1/4*(1-xi)*(1+eta)*(1-Si**2)
        shape[16, 0] = 1/4*(1-xi**2)*(1-eta)*(1+Si)
        shape[17, 0] = 1/4*(1+xi)*(1-eta**2)*(1+Si)
        shape[18, 0] = 1/4*(1-xi**2)*(1+eta)*(1+Si)
        shape[19, 0] = 1/4*(1-xi)*(1-eta**2)*(1+Si)

        return shape
