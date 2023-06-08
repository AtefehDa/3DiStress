
import math

import numpy as np
import pandas as pd

np.set_printoptions(precision=4)

class globalstiffnesscalculation:
    def __init__(self, PD, NPE, NoE, NoN, GPE, E_elm, nu_elm):
        self.PD = PD
        self.NPE = NPE
        self.NoE = NoE
        self.NoN = NoN
        self.GPE = GPE
        self.E_elm = E_elm
        self.nu_elm = nu_elm

    def globalstiffness_matrix(self, ENL, EL, NL_Cartesian):
        K_global = np.zeros([self.NoN*self.PD, self.NoN*self.PD])
        for i in range(1, self.NoE+1):
            nl = EL[i-1, 0:self.NPE]
            nl = nl.astype(int)
            EL = EL.astype(int)
            E = self.E_elm[i-1]
            nu = self.nu_elm[i-1]

            kelementnew = globalstiffnesscalculation.newelementstiffness(self, nl, NL_Cartesian, E, nu)

            for r in range(0, self.NPE):
                for p in range(0, self.PD):
                    for q in range(0, self.NPE):
                        for s in range(0, self.PD):

                            row = ENL[nl[r]-1, p+3*self.PD]
                            column = ENL[nl[q]-1, s+3*self.PD]

                            row = int(row)
                            column = int(column)

                            value = kelementnew[r*self.PD+p, q*self.PD+s]
                            K_global[row-1, column-1] = K_global[row-1, column-1] + value
        return K_global

    def newelementstiffness(self, nl, NL_Cartesian, E, nu):
        x = np.zeros([self.NPE, self.PD])
        x[0:self.NPE, 0:self.PD] = NL_Cartesian[nl[0:self.NPE]-1, 0:self.PD]
        kelementnew = np.zeros([self.NPE*self.PD, self.NPE*self.PD])
        trans_x = x.T
        for gp in range(1, self.GPE+1):
            Jacobian = np.zeros([self.PD, self.PD])
            grad_integral = np.zeros([self.PD, self.NPE])
            (xi, eta, Si, alpha) = globalstiffnesscalculation.Gausspoint(self, gp)
            derivative = globalstiffnesscalculation.grad_shapefunction(self, xi, eta, Si)
            Jacobian = trans_x @ derivative.T
            detj = (np.linalg.det(Jacobian))
            grad_integral = np.linalg.inv(Jacobian).T @ derivative
            C = globalstiffnesscalculation.constitutive(self, E, nu)
            bb = np.zeros([6, self.PD])
            B = np.zeros([6, self.PD*self.NPE])
            for i in range(1, self.NPE+1):
                bb[0, 0] = bb[3, 1] = bb[5, 2] = grad_integral[0, i-1]
                bb[1, 1] = bb[3, 0] = bb[4, 2] = grad_integral[1, i-1]
                bb[2, 2] = bb[4, 1] = bb[5, 0] = grad_integral[2, i-1]
                B[:, (i-1)*3: (i-1)*3+3] = bb

            kelementnew = kelementnew + (B.T @ C @ B)* detj * alpha
        return kelementnew

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

    def constitutive(self, E, nu):

        C = np.zeros([6, 6])
        lam = ((E)/((1+nu)*(1-2*nu)))
        C[0, 0] = C[1, 1] = C[2, 2] = (1-nu)*lam
        C[0, 1] = C[0, 2] = C[1, 0] = C[1, 2] = C[2, 1] = C[2, 0] = nu*lam
        C[3, 3] = C[4, 4] = C[5, 5] = (1-2*nu)*lam

        return C

    def shapefunction(self, xi, eta, Si):
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

