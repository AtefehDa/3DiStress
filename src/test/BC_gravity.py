import math

import numpy as np
import pandas as pd

class Boundaryconditionsassignment:
    def __init__(self, PD, NoN, NPE, NoE, EL, m, p, k, sigmav, dispp, disssp, max_y, min_x, min_y, max_x, E_node):

        self.PD = PD
        self.NoN = NoN
        self.NPE = NPE
        self.NoE = NoE
        self.EL = EL
        self.m = m
        self.p = p
        self.k = k
        self.sigmav = sigmav
        self.dispp = dispp
        self.disssp = disssp
        self.max_y = max_y
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.E_node = E_node

    def BC(self, NL_Cartesian):

        ENL = np.zeros([self.NoN, 6*self.PD])
        ENL[:, 0:self.PD] = NL_Cartesian
        ENL[:, 3] = ENL[:, 4] = ENL[:, 5] = 1
######################bottom face#################
        dirichlet_nodes = np.zeros([self.m*self.p, 8])
        for i in range(0, self.m*self.p):
            dirichlet_nodes[i, 0:4] = self.EL[i, 0:4]
            dirichlet_nodes[i, 4:] = self.EL[i, 8:12]
        dirichlet_nodes = dirichlet_nodes.astype(int)
############# sigma H back face############
        kk = 0
        elmlist_b = np.zeros([self.p * self.k, 8])
        for ki in range(0, self.k):
            for i in range((self.m-1)*(self.p)+ki*self.m*self.p, self.m*self.p*(ki+1)):
                elmlist_b[kk, :] = self.EL[i, [3, 10, 2, 14, 6, 18, 7, 15]]
                kk += 1
        elmlist_b = elmlist_b.astype(int)
        elmlist_b = np.unique(elmlist_b.flatten())
        for i in elmlist_b:
            ENL[i-1, 3] = -1
            ENL[i-1, 4] = -1
            ENL[i-1, 13] = -self.dispp*0.5
            ENL[i-1, 12] = (self.disssp/10)*0.5
# # ################sigma H front face##########
        jj = 0
        elmlist_f = np.zeros([self.p * self.k, 8])
        for ki in range(0, self.k):
            for i in range(ki*self.m*self.p, ki*self.m*self.p+self.p):
                elmlist_f[jj, :] = self.EL[i, [0, 8, 1, 13, 5, 16, 4, 12]]
                jj += 1
        elmlist_f = elmlist_f.astype(int)
        elmlist_f = np.unique(elmlist_f.flatten())
        for i in elmlist_f:
            ENL[i-1, 3] = -1
            ENL[i-1, 4] = -1
            ENL[i-1, 13] = self.dispp*0.5
            ENL[i-1, 12] = -(self.disssp/10)*0.5
###################sigma h left face########
        ii = 0
        elmlist_l = np.zeros([self.m * self.k, 8])
        for ki in range(0, self.k):
            for i in range(ki*self.m*self.p+1, (ki+1)*self.m*self.p+1, self.p):
                elmlist_l[ii, :] = self.EL[i-1, [0, 11, 3, 15, 7, 19, 4, 12]]
                ii += 1
        elmlist_l = elmlist_l.astype(int)
        elmlist_l = np.unique(elmlist_l.flatten())
        for i in elmlist_l:
            ENL[i-1, 3] = -1
            ENL[i-1, 4] = -1
            ENL[i-1, 13] = -(self.disssp/10)*0.5 + ENL[i-1, 13]
            ENL[i-1, 12] = self.dispp*0.1 +  ENL[i-1, 12]
# # #################sigma h right face########
        ff = 0
        elmlist_r = np.zeros([self.m * self.k, 8])
        for ki in range(0, self.k):
            for i in range(ki*self.m*self.p+self.p, (ki+1)*self.m*self.p+1, self.p):
                elmlist_r[ff, :] = self.EL[i-1, [1, 9, 2, 14, 6, 17, 5, 13]]
                ff += 1
        elmlist_r = elmlist_r.astype(int)
        elmlist_r = np.unique(elmlist_r.flatten())
        for i in elmlist_r:
            ENL[i-1, 3] = -1
            ENL[i-1, 4] = -1
            ENL[i-1, 13] = (self.disssp/10)*0.5 + ENL[i-1, 13]
            ENL[i-1, 12] = -self.dispp*0.1 + ENL[i-1, 12]
#################sigma vertical ########
        j = 0
        elmlist_v = np.zeros([self.p * self.m, 8])
        for i in range(3*self.m*self.p, self.m*self.p*self.k):
            elmlist_v[j, :] = self.EL[i, [4, 16, 5, 17, 6, 18, 7, 19]]
            j += 1
        elmlist_v = elmlist_v.astype(int)
############defining boundary conditions##########
        dirichlet_nodes = np.unique(dirichlet_nodes.flatten())
        for i in dirichlet_nodes:
            ENL[i-1, 5] = -1
        DOFS = 0
        DOCS = 0
        for i in range(0, self.NoN):
            for j in range(0, self.PD):

                if ENL[i, self.PD+j] == -1 :
                    DOCS -= 1
                    ENL[i, 2*self.PD+j] = DOCS
                else:
                    DOFS += 1
                    ENL[i, 2*self.PD+j] = DOFS

        for i in range(0, self.NoN):
            for j in range(0, self.PD):

                if ENL[i, 2*self.PD+j] < 0 :
                    ENL[i, 3*self.PD+j] = abs(ENL[i, 2*self.PD+j]) + DOFS
                else:
                    ENL[i, 3*self.PD+j] = abs(ENL[i, 2*self.PD+j])
                    DOCS = abs(DOCS)
        print(f"DOCS is {DOCS}")

######################vertical load application####################################################
        z_gradient = np.zeros([8, 1])
        for i in range(0, np.size(elmlist_v, 0)):
            coordinate_one_element_v = ENL[elmlist_v[i, :]-1, 0:self.PD]
            Nodal_force_v = np.zeros([8*self.PD, 1])
            traction_force_v = np.zeros([self.PD, 1])
            nod_force_v = np.zeros([8, (self.PD+1)])
            for g in range(1, 4+1):
                (x_bou, eta_bou, alpaha_bou) = Boundaryconditionsassignment.gausspoint_boundary(self, g)
                derivative_boundary = Boundaryconditionsassignment.grad_shapefunction(self, x_bou, eta_bou)
                Jacobian_3d = (coordinate_one_element_v[:,[0,1]].T) @ (derivative_boundary.T)
                det_jacobian = (np.linalg.det(Jacobian_3d))
                shape_function = Boundaryconditionsassignment.gausspoint_boundary_traction(self, x_bou, eta_bou)
                traction_force_v[0, 0] = 0
                traction_force_v[1, 0] = 0
                traction_force_v[2, 0] = -self.sigmav
                Nodal_force_v = Nodal_force_v + ((shape_function @ traction_force_v) * (det_jacobian ) * alpaha_bou)
            nod_force_v[:,1:4] = Nodal_force_v.reshape(8, 3)

            for ii in range(0, 8):
                nod_force_v[ii, 0] = elmlist_v[i, ii]
                z_gradient[ii, 0] = (460.908 - coordinate_one_element_v[ii, 2])/1000
                nod_force_v[ii, 3] = nod_force_v[ii, 3] * z_gradient[ii, 0]
                ENL[int(nod_force_v[ii, 0])-1, 5*self.PD+2] = nod_force_v[ii, 3] + ENL[int(nod_force_v[ii, 0])-1, 5*self.PD+2]
#####################################APPLICATION OF GRAVITY #################################################################
        density = np.loadtxt("src/data/DEN_ELEM.txt", dtype=float)
        for i in range(0, np.size(self.EL, 0)):
            coordinate_one_element_g = ENL[self.EL[i, :]-1, 0:self.PD]
            Nodal_force_g = np.zeros([self.NPE*self.PD, 1])
            gravity = np.zeros([self.PD, 1])
            nod_force_g = np.zeros([self.NPE, (self.PD+1)])
            for g in range(1, 8+1):
                (xi, eta, Si, alpha) = Boundaryconditionsassignment.Gausspoint(self, g)
                derivative = Boundaryconditionsassignment.grad_shapefunction_gravity(self, xi, eta, Si)
                Jacobian_3d_g = (coordinate_one_element_g.T) @ (derivative.T)
                detjdens = (np.linalg.det(Jacobian_3d_g))
                shape_function = Boundaryconditionsassignment.gausspoint_boundary_gravity(self, xi, eta, Si)
                gravity[0, 0] = 0
                gravity[1, 0] = 0
                gravity[2, 0] = -(density[i] * 10)           #pascal
                Nodal_force_g = Nodal_force_g + ((shape_function @ gravity) * detjdens * alpha)
            nod_force_g[:,1:4] = Nodal_force_g.reshape(self.NPE, 3)
            for ii in range(0, self.NPE):
                nod_force_g[ii, 0] = self.EL[i, ii]
                ENL[int(nod_force_g[ii, 0])-1, 5*self.PD+2] = ENL[int(nod_force_g[ii, 0])-1, 5*self.PD+2] + nod_force_g[ii, 3]

        return ENL, DOFS, DOCS

    def gausspoint_boundary(self, g):

        if g == 1:
            x_bou = -1/math.sqrt(3)
            eta_bou = -1/math.sqrt(3)
            alpaha_bou = 1
        if g == 2:
            x_bou = 1/math.sqrt(3)
            eta_bou = -1/math.sqrt(3)
            alpaha_bou = 1
        if g == 3:
            x_bou = 1/math.sqrt(3)
            eta_bou = 1/math.sqrt(3)
            alpaha_bou = 1
        if g == 4:
            x_bou = -1/math.sqrt(3)
            eta_bou = 1/math.sqrt(3)
            alpaha_bou = 1

        return x_bou, eta_bou, alpaha_bou

    def gausspoint_boundary_coordinate(self, x_bou, eta_bou):

        shape_function_coor = np.zeros([1, 8])

        shape_function_coor[0, 0] = -1/4 * (1 - x_bou) * (1 - eta_bou) * (1 + x_bou + eta_bou)
        shape_function_coor[0, 1] =  1/2 * (1 - x_bou) * (1 + x_bou) * (1 - eta_bou)
        shape_function_coor[0, 2] = -1/4 * (1 + x_bou) * (1 - eta_bou) * (1 - x_bou + eta_bou)
        shape_function_coor[0, 3] = 1/2 * (1 + x_bou) * (1 + eta_bou) * (1 - eta_bou)
        shape_function_coor[0, 4] = -1/4 * (1 + x_bou) * (1 + eta_bou) * (1 - x_bou - eta_bou)
        shape_function_coor[0, 5] =  1/2 * (1 - x_bou) * (1 + x_bou) * (1 + eta_bou)
        shape_function_coor[0, 6] = -1/4 * (1 - x_bou) * (1 + eta_bou) * (1 + x_bou - eta_bou)
        shape_function_coor[0, 7] = 1/2 * (1 - x_bou) * (1 + eta_bou) * (1 - eta_bou)

        return shape_function_coor

    def grad_shapefunction(self, x_bou, eta_bou):
        derivative_boundary = np.zeros([2, 8])
        derivative_boundary[0, 0] = 1/4 * (1 - eta_bou) * (1 + x_bou + eta_bou) - 1/4 * (1 - x_bou) * (1 - eta_bou)
        derivative_boundary[0, 1] = -x_bou * (1 - eta_bou)
        derivative_boundary[0, 2] = -1/4 * (1 - eta_bou) * (1 - x_bou + eta_bou) + 1/4 * (1 + x_bou) * (1 - eta_bou)
        derivative_boundary[0, 3] = 1/2 * (1 + eta_bou) * (1 - eta_bou)
        derivative_boundary[0, 4] = -1/4 * (1 + eta_bou) * (1 - x_bou - eta_bou) + 1/4 * (1 + x_bou) * (1 + eta_bou)
        derivative_boundary[0, 5] = -x_bou * (1 + eta_bou)
        derivative_boundary[0, 6] = 1/4 * (1 + eta_bou) * (1 + x_bou - eta_bou) - 1/4 * (1 - x_bou) * (1 + eta_bou)
        derivative_boundary[0, 7] = -1/2 * (1 + eta_bou) * (1 - eta_bou)

        derivative_boundary[1, 0] = 1/4 * (1 - x_bou)*(1 + x_bou + eta_bou) - 1/4 * (1 - x_bou) * (1 - eta_bou)
        derivative_boundary[1, 1] = -1/2 * (1 - x_bou) * (1 + x_bou)
        derivative_boundary[1, 2] = 1/4 * (1 + x_bou)* (1 - x_bou + eta_bou) - 1/4 * (1 + x_bou) * (1 - eta_bou)
        derivative_boundary[1, 3] = -eta_bou * (1 + x_bou)
        derivative_boundary[1, 4] = -1/4 * (1 + x_bou)* (1 - x_bou - eta_bou) + 1/4 * (1 + x_bou) * (1 + eta_bou)
        derivative_boundary[1, 5] = 1/2 * (1 - x_bou) * (1 + x_bou)
        derivative_boundary[1, 6] = -1/4 * (1 - x_bou)*(1 + x_bou - eta_bou) + 1/4 * (1 - x_bou) * (1 + eta_bou)
        derivative_boundary[1, 7] = -eta_bou * (1 - x_bou)

        return derivative_boundary

    def gausspoint_boundary_traction(self, x_bou, eta_bou):

        shape_function = np.zeros([24, self.PD])

        shape_function[0, 0] = shape_function[1, 1] = shape_function[2, 2] = -1/4 * (1 - x_bou) * (1 - eta_bou) * (1 + x_bou + eta_bou)
        shape_function[3, 0] = shape_function[4, 1] = shape_function[5, 2]  =  1/2 * (1 - x_bou) * (1 + x_bou) * (1 - eta_bou)
        shape_function[6, 0] = shape_function[7, 1] = shape_function[8, 2]  = -1/4 * (1 + x_bou) * (1 - eta_bou) * (1 - x_bou + eta_bou)
        shape_function[9, 0] = shape_function[10, 1] = shape_function[11, 2]  = 1/2 * (1 + x_bou) * (1 + eta_bou) * (1 - eta_bou)
        shape_function[12, 0] = shape_function[13, 1] = shape_function[14, 2] = -1/4 * (1 + x_bou) * (1 + eta_bou) * (1 - x_bou - eta_bou)
        shape_function[15, 0] = shape_function[16, 1] = shape_function[17, 2]  = 1/2 * (1 - x_bou) * (1 + x_bou) * (1 + eta_bou)
        shape_function[18, 0] = shape_function[19, 1] = shape_function[20, 2]  = -1/4 * (1 - x_bou) * (1 + eta_bou) * (1 + x_bou - eta_bou)
        shape_function[21, 0] = shape_function[22, 1] = shape_function[23, 2]  = 1/2 * (1 - x_bou) * (1 + eta_bou) * (1 - eta_bou)

        return shape_function

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

    def grad_shapefunction_gravity(self, xi, eta, Si):
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

    def shapefunction_gravity(self, xi, eta, Si):
        shape_gravity = np.zeros([self.NPE, 1])

        shape_gravity[0, 0] = 1/8*(1-xi)*(1-eta)*(1-Si)*(-xi-eta-Si-2)
        shape_gravity[1, 0] = 1/8*(1+xi)*(1-eta)*(1-Si)*(xi-eta-Si-2)
        shape_gravity[2, 0] = 1/8*(1+xi)*(1+eta)*(1-Si)*(xi+eta-Si-2)
        shape_gravity[3, 0] = 1/8*(1-xi)*(1+eta)*(1-Si)*(-xi+eta-Si-2)
        shape_gravity[4, 0] = 1/8*(1-xi)*(1-eta)*(1+Si)*(-xi-eta+Si-2)
        shape_gravity[5, 0] = 1/8*(1+xi)*(1-eta)*(1+Si)*(xi-eta+Si-2)
        shape_gravity[6, 0] = 1/8*(1+xi)*(1+eta)*(1+Si)*(xi+eta+Si-2)
        shape_gravity[7, 0] = 1/8*(1-xi)*(1+eta)*(1+Si)*(-xi+eta+Si-2)
        shape_gravity[8, 0] = 1/4*(1-xi**2)*(1-eta)*(1-Si)
        shape_gravity[9, 0] = 1/4*(1+xi)*(1-eta**2)*(1-Si)
        shape_gravity[10, 0] = 1/4*(1-xi**2)*(1+eta)*(1-Si)
        shape_gravity[11, 0] = 1/4*(1-xi)*(1-eta**2)*(1-Si)
        shape_gravity[12, 0] = 1/4*(1-xi)*(1-eta)*(1-Si**2)
        shape_gravity[13, 0] = 1/4*(1+xi)*(1-eta)*(1-Si**2)
        shape_gravity[14, 0] = 1/4*(1+xi)*(1+eta)*(1-Si**2)
        shape_gravity[15, 0] = 1/4*(1-xi)*(1+eta)*(1-Si**2)
        shape_gravity[16, 0] = 1/4*(1-xi**2)*(1-eta)*(1+Si)
        shape_gravity[17, 0] = 1/4*(1+xi)*(1-eta**2)*(1+Si)
        shape_gravity[18, 0] = 1/4*(1-xi**2)*(1+eta)*(1+Si)
        shape_gravity[19, 0] = 1/4*(1-xi)*(1-eta**2)*(1+Si)

        return shape_gravity

    def gausspoint_boundary_gravity(self, xi, eta, Si):

        shape_function = np.zeros([self.PD*self.NPE, self.PD])

        shape_function[0, 0] = shape_function[1, 1] = shape_function[2, 2] = 1/8*(1-xi)*(1-eta)*(1-Si)*(-xi-eta-Si-2)
        shape_function[3, 0] = shape_function[4, 1] = shape_function[5, 2]  =  1/8*(1+xi)*(1-eta)*(1-Si)*(xi-eta-Si-2)
        shape_function[6, 0] = shape_function[7, 1] = shape_function[8, 2]  = 1/8*(1+xi)*(1+eta)*(1-Si)*(xi+eta-Si-2)
        shape_function[9, 0] = shape_function[10, 1] = shape_function[11, 2]  = 1/8*(1-xi)*(1+eta)*(1-Si)*(-xi+eta-Si-2)
        shape_function[12, 0] = shape_function[13, 1] = shape_function[14, 2] = 1/8*(1-xi)*(1-eta)*(1+Si)*(-xi-eta+Si-2)
        shape_function[15, 0] = shape_function[16, 1] = shape_function[17, 2]  = 1/8*(1+xi)*(1-eta)*(1+Si)*(xi-eta+Si-2)
        shape_function[18, 0] = shape_function[19, 1] = shape_function[20, 2]  = 1/8*(1+xi)*(1+eta)*(1+Si)*(xi+eta+Si-2)
        shape_function[21, 0] = shape_function[22, 1] = shape_function[23, 2]  = 1/8*(1-xi)*(1+eta)*(1+Si)*(-xi+eta+Si-2)
        shape_function[24, 0] = shape_function[25, 1] = shape_function[26, 2]  = 1/4*(1-xi**2)*(1-eta)*(1-Si)
        shape_function[27, 0] = shape_function[28, 1] = shape_function[29, 2]  = 1/4*(1+xi)*(1-eta**2)*(1-Si)
        shape_function[30, 0] = shape_function[31, 1] = shape_function[32, 2]  = 1/4*(1-xi**2)*(1+eta)*(1-Si)
        shape_function[33, 0] = shape_function[34, 1] = shape_function[35, 2]  = 1/4*(1-xi)*(1-eta**2)*(1-Si)
        shape_function[36, 0] = shape_function[37, 1] = shape_function[38, 2]  = 1/4*(1-xi)*(1-eta)*(1-Si**2)
        shape_function[39, 0] = shape_function[40, 1] = shape_function[41, 2]  = 1/4*(1+xi)*(1-eta)*(1-Si**2)
        shape_function[42, 0] = shape_function[43, 1] = shape_function[44, 2]  = 1/4*(1+xi)*(1+eta)*(1-Si**2)
        shape_function[45, 0] = shape_function[46, 1] = shape_function[47, 2]  = 1/4*(1-xi)*(1+eta)*(1-Si**2)
        shape_function[48, 0] = shape_function[49, 1] = shape_function[50, 2]  = 1/4*(1-xi**2)*(1-eta)*(1+Si)
        shape_function[51, 0] = shape_function[52, 1] = shape_function[53, 2]  = 1/4*(1+xi)*(1-eta**2)*(1+Si)
        shape_function[54, 0] = shape_function[55, 1] = shape_function[56, 2]  = 1/4*(1-xi**2)*(1+eta)*(1+Si)
        shape_function[57, 0] = shape_function[58, 1] = shape_function[59, 2]  = 1/4*(1-xi)*(1-eta**2)*(1+Si)

        return shape_function