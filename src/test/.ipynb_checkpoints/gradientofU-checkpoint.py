import numpy as np
import math

class DoingPostprocess:

    def __init__(self, PD, NoE, NPE, GPE, E, nu):
        self.PD = PD
        self.NoE = NoE
        self.NPE = NPE
        self.GPE = GPE
        self.E = E
        self.nu = nu
    def postprocess(self, NL, EL, ENL):

        self.PD = np.size(NL, 1)
        self.NoE = np.size(EL, 0)
        self.NPE = np.size(EL, 1)
        scale = 5                         #magnify the results

        disp, strain, stress = DoingPostprocess.element_postprocess(self, NL, EL, ENL)
        strain_xx= np.zeros([self.NPE, self.NoE])
        strain_xy= np.zeros([self.NPE, self.NoE])
        strain_yx= np.zeros([self.NPE, self.NoE])
        strain_yy= np.zeros([self.NPE, self.NoE])
        stress_xx = np.zeros([self.NPE, self.NoE])                     #In fact this should be in the order of ([GPE, NOE]), here NPE = GPE
        stress_xy = np.zeros([self.NPE, self.NoE])
        stress_yx= np.zeros([self.NPE, self.NoE])
        stress_yy= np.zeros([self.NPE, self.NoE])

        dispx = np.zeros([self.NPE, self.NoE])
        dispy = np.zeros([self.NPE, self.NoE])

        xnew = np.zeros([self.NPE, self.NoE])
        ynew = np.zeros([self.NPE, self.NoE])

        if self.NPE in [3,4]:

            xnew = ENL[EL-1, 0] + scale * ENL[EL-1, 4*self.PD]
            ynew = ENL[EL-1, 1] + scale * ENL[EL-1, 4*self.PD+1]

            xnew = xnew.T
            ynew = ynew.T

            strain_xx[:, :] = strain[:, :, 0, 0].T
            strain_xy[:, :] = strain[:, :, 0, 1].T
            strain_yx[:, :] = strain[:, :, 1, 0].T
            strain_yy[:, :] = strain[:, :, 1, 1].T
            stress_xx[:, :] = stress[:, :, 0, 0].T
            stress_xy[:, :] = stress[:, :, 0, 1].T
            stress_yx[:, :] = stress[:, :, 1, 0].T
            stress_yy[:, :] = stress[:, :, 1, 1].T

            dispx = disp[:, :, 0, 0].T
            dispy = disp[:, :, 1, 0].T


        return (stress_xx, stress_xy, stress_yx, stress_yy, strain_xx, strain_xy,
                strain_yx, strain_yy, dispx, dispy, xnew, ynew)

    def element_postprocess(self, NL, EL, ENL):
        self.PD = np.size(NL, 1)
        self.NoE = np.size(EL, 0)
        self.NPE = np.size(EL, 1)
        self.NoN = np.size(NL, 0)

        if self.NPE == 3 :
            GPE = 1
        if self.NPE == 4:
            GPE = 4

        disp = np.zeros([self.NoE, self.NPE, self.PD, 1])                                  #calculated on nodes
        stress = np.zeros([self.NoE, GPE, self.PD, self.PD])
        strain = np.zeros([self.NoE, GPE, self.PD, self.PD])

        for e in range(0, self.NoE):
            nl = EL[e, 0:self.NPE]
            for i in range(0, self.NPE):
                for j in range(0, self.PD):
                    disp[e, i, j, 0] = ENL[nl[i]-1, 4*self.PD+j]                       #displacement on nodes ##First loop over elements

        Num_of_patches = 0
        for NODE in range(1, self.NoN):
            patch = np.zeros([self.NoE, self.NPE])
            count = 0
            A_patch = 0
            bsmall1 = 0
            bsmall2 = 0
            bsmall3 = 0
            bsmall4 = 0
            B1 = 0
            for j in range(0, self.NoE):
                nl = EL[j, 0:self.NPE]
                nl = nl.astype(int)
                if NODE in nl:
                    count+=1
                    if count == 4:
                        print(NODE)
                        Num_of_patches += 1
                        for J in range(0, self.NoE):
                            if NODE in EL[J, 0:self.NPE]:
                                patch[J, 0:self.NPE] = EL[J, 0:self.NPE]

                        NONZEROterms = patch[np.where(patch!=0)]
                        connected_el = NONZEROterms.reshape([self.NPE, self.NPE])
                        print(f"connected_el is \n {connected_el}")

                        for k in range(0, np.size(connected_el, 0)):
                            nlnew = connected_el[k, 0:self.NPE]
                            nlnew = nlnew.astype(int)

                            xn = np.zeros([self.NPE, self.PD])                                   #specify the corners of the elements
                            xn[0:self.NPE, 0:self.PD] = NL[nlnew[0:self.NPE]-1, 0:self.PD]

                            u = np.zeros([self.NPE, self.PD])                                    #somehow defined the displacement on corners

                            for i in range(0, self.NPE):
                                for j in range(0, self.PD):
                                    u[i, j] = ENL[nlnew[i]-1, 4*self.PD+j]                        #@ EACH SUPERCONVERGENT POINT = GAUSS POINT

                            trans_x = xn.T

                            for gp in range(1, GPE+1):

                                Jacobian = np.zeros([self.PD, self.PD])
                                grad_integral = np.zeros([self.PD, self.NPE])

                                (xi, eta, alpha) = DoingPostprocess.Gausspoint(self, self.NPE, GPE, gp)
                                derivative = DoingPostprocess.grad_shapefunction(self, self.NPE, xi, eta)
                                shapefunction = DoingPostprocess.shape_functions(self, self.NPE, xi, eta)
                                Jacobian = trans_x @ derivative.T
                                grad_integral = np.linalg.inv(Jacobian).T @ derivative

                                g =  grad_integral @ u                                 #each gauss point has a 2*2 gradient matrix : approximation of displacement
                                                                                       #each gp has a g matrix which is the displacement gradient at gp
                                X_Yk = shapefunction.T @ xn                             ##approximation ofcoordinates of gauss point in physical domain
                                q = np.zeros([1,6])
                                q[0, 0] = 1
                                q[0, 1] = X_Yk[0, 0]
                                q[0, 2] = X_Yk[0, 1]
                                q[0, 3] = X_Yk[0, 0] * X_Yk[0, 1]
                                q[0, 4] = X_Yk[0, 0] ** 2
                                q[0, 5] = X_Yk[0, 1] ** 2
                                q_trans = q.T
                                B1 = q_trans @ q
                                A_patch = A_patch + B1

                                bsmall1 = bsmall1 + q_trans * g[0, 0]
                                bsmall2 = bsmall2 + q_trans * g[0, 1]
                                bsmall3 = bsmall3 + q_trans * g[1, 0]
                                bsmall4 = bsmall4 + q_trans * g[1, 1]

                        print(f"A_patch is \n {A_patch}")                                                       #in each patch you earned the A_patch matrix

                        asmall1 = np.linalg.inv(A_patch) @ bsmall1
                        asmall2 = np.linalg.inv(A_patch) @ bsmall2
                        asmall3 = np.linalg.inv(A_patch) @ bsmall3
                        asmall4 = np.linalg.inv(A_patch) @ bsmall4

                        COOR_CENTER = NL[NODE-1, 0:self.PD]
                        solution_center = np.array([1, COOR_CENTER[0], COOR_CENTER[1],  COOR_CENTER[0]*COOR_CENTER[1], COOR_CENTER[0]**2, COOR_CENTER[1]**2])
                        ucenter = np.zeros([1, 4])
                        ucenter[0, 0] = solution_center @ asmall1
                        ucenter[0, 1] = solution_center @ asmall2
                        ucenter[0, 2] = solution_center @ asmall3
                        ucenter[0, 3] = solution_center @ asmall4

                        listofsidenodes = []
                        uboundary = np.zeros([1, 4])

                        for node in range(1, self.NoN+1):
                            occurrences = np.count_nonzero(EL == node)
                            if occurrences == 1:
                                if node in connected_el:
                                    bound_node = node
                                    print(f" {bound_node} repeated {occurrences} times")
                                    COOR_BOUND = NL[bound_node-1, 0:self.PD]
                                    solution_bound_node = np.array([1, COOR_BOUND[0], COOR_BOUND[1],  COOR_BOUND[0]*COOR_BOUND[1], COOR_BOUND[0]**2, COOR_BOUND[1]**2])
                                    uboundary[0, 0] = solution_bound_node @ asmall1
                                    uboundary[0, 1] = solution_bound_node @ asmall2
                                    uboundary[0, 2] = solution_bound_node @ asmall3
                                    uboundary[0, 3] = solution_bound_node @ asmall4
                                    print(f"{bound_node} has the nodal values as \n {uboundary}")

                            if 1 < occurrences < 4:
                                if node in connected_el:
                                    sidenode = listofsidenodes.append(node)
                        print(f"side_node is \n {listofsidenodes}")

                        uside = np.zeros([len(listofsidenodes), 4])
                        for l in range(0, len(listofsidenodes)):
                            side_node = listofsidenodes[l]
                            COOR_SIDE = NL[side_node-1, 0:self.PD]
                            solution_side_node = np.array([1, COOR_SIDE[0], COOR_SIDE[1],  COOR_SIDE[0]*COOR_SIDE[1], COOR_SIDE[0]**2, COOR_SIDE[1]**2])

                            uside[l, 0] = solution_side_node  @ asmall1
                            uside[l, 1] = solution_side_node  @ asmall2
                            uside[l, 2] = solution_side_node  @ asmall3
                            uside[l, 3] = solution_side_node  @ asmall4


                        print(f"{NODE} has the nodal values as \n {ucenter} ")
                        print(f"uside has the nodal values as \n {uside}")
                        print("*"*100)

        print(f"number of patches is \n {Num_of_patches}")



        return ucenter, uboundary, uside


    def Gausspoint(self, NPE, GPE, gp):

        if NPE == 4 :
            if GPE == 1:
                if gp == 1:
                    xi = 0
                    eta = 0
                    alpha = 4
            if GPE == 4 :
                if gp == 1:
                    xi = -1/math.sqrt(3)
                    eta = -1/math.sqrt(3)
                    alpha = 1
                if gp == 2:
                    xi = 1/math.sqrt(3)
                    eta = -1/math.sqrt(3)
                    alpha = 1
                if gp == 3:
                    xi = 1/math.sqrt(3)
                    eta = 1/math.sqrt(3)
                    alpha = 1
                if gp == 4:
                    xi = -1/math.sqrt(3)
                    eta = 1/math.sqrt(3)
                    alpha = 1

        return (xi, eta, alpha)


    def grad_shapefunction(self, NPE, xi, eta):

        self.PD = 2
        self.NPE = 4
        derivative = np.zeros([self.PD, NPE])
        if self.NPE == 4:
            derivative[0, 0] = -1/4*(1-eta)
            derivative[0, 1] = 1/4*(1-eta)
            derivative[0, 2] = 1/4*(1+eta)
            derivative[0, 3] = -1/4*(1+eta)

            derivative[1, 0] = -1/4*(1-xi)
            derivative[1, 1] = -1/4*(1+xi)
            derivative[1, 2] = 1/4*(1+xi)
            derivative[1, 3] = 1/4*(1-xi)
        if NPE == 8:
            pass
        return derivative

    def dyad(self, u, v):

        u = u.reshape(len(u), 1)
        v = v.reshape(len(v), 1)
        PD = 2
        A = u @ v.T

        return A


    def constitutive(self, i, j, k, l, E, nu):
        c = (E/(2*(1+nu))) * (DoingPostprocess.delta(self, i, l) * DoingPostprocess.delta(self, j, k) + \
        DoingPostprocess.delta(self, i, k)*DoingPostprocess.delta(self, j, l)) + ((E*nu)/(1-nu**2)) * DoingPostprocess.delta(self, i, j) *\
        DoingPostprocess.delta(self, k, l)
        return c

    def delta(self, a, b):
        if a == b :
            delta = 1
        else:
            delta = 0
        return delta

    def shape_functions(self, NPE, xi, eta):

        shapefunction = np.zeros([NPE,1])
        shapefunction[0, 0] = 1/4 * (1-xi) * (1-eta)
        shapefunction[1, 0] = 1/4 * (1+xi) * (1-eta)
        shapefunction[2, 0] = 1/4 * (1+xi) * (1+eta)
        shapefunction[3, 0] = 1/4 * (1-xi) * (1+eta)

        return shapefunction

