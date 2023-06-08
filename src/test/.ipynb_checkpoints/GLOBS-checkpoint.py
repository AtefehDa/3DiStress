import numpy as np
import math

class globalstiffnesscalculation:
    def __init__(self, PD, NPE, NoE, NoN, GPE, E, nu):
        self.PD = PD
        self.NPE = NPE
        self.NoE = NoE
        self.NoN = NoN
        self.GPE = GPE
        self.E = E
        self.nu = nu

    def globalstiffness_matrix(self, ENL, EL, NL):
        K_global = np.zeros([self.NoN*self.PD, self.NoN*self.PD])
        for i in range(1, self.NoE+1):
            nl = EL[i-1, 0:self.NPE]          #one_dimensional_array
            nl = nl.astype(int)
            EL = EL.astype(int)

            kelementnew = globalstiffnesscalculation.newelementstiffness(self, nl, NL)

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

    def newelementstiffness(self, nl, NL):
        nl = nl.astype(int)
        x = np.zeros([self.NPE,self.PD])
        x[0:self.NPE, 0:self.PD] = NL[nl[0:self.NPE]-1, 0:self.PD]
        kelementnew = np.zeros([self.NPE*self.PD, self.NPE*self.PD])
        trans_x = x.T

        if self.NPE == 3:
            GPE = 1

        if self.NPE == 4:
            GPE = 4

        for i in range(1, self.NPE+1):
            for j in range(1, self.NPE+1):
                k_small = np.zeros([self.PD, self.PD])
                for gp in range(1, GPE+1):
                    Jacobian = np.zeros([self.PD, self.PD])
                    grad_integral = np.zeros([self.PD, self.NPE])
                    (xi, eta, alpha) = globalstiffnesscalculation.Gausspoint(self, gp)
                    derivative = globalstiffnesscalculation.grad_shapefunction(self, xi, eta)
                    Jacobian = trans_x @ derivative.T
                    grad_integral = np.linalg.inv(Jacobian).T @ derivative
                    for a in range(1,self.PD+1):
                        for c in range(1,self.PD+1):
                            for b in range(1,self.PD+1):
                                for d in range(1,self.PD+1):
                                    k_small[a-1, c-1] =  k_small[a-1, c-1] + (grad_integral[b-1, i-1] * \
                                    globalstiffnesscalculation.constitutive(self, a, b, c, d) * grad_integral[d-1, j-1]) * \
                                    (np.linalg.det(Jacobian) * alpha)
                    kelementnew[((i-1)*self.PD+1)-1 : i*self.PD, ((j-1)*self.PD+1)-1: j*self.PD] = k_small
        return kelementnew

    def Gausspoint(self, gp):
        if self.NPE == 4 :

            if self.GPE == 1:

                if gp == 1:

                    xi = 0
                    eta = 0
                    alpha = 4
            if self.GPE == 4 :
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


    def grad_shapefunction(self, xi, eta):

        derivative = np.zeros([self.PD, self.NPE])
        if self.NPE == 4:
            derivative[0, 0] = -1/4*(1-eta)
            derivative[0, 1] = 1/4*(1-eta)
            derivative[0, 2] = 1/4*(1+eta)
            derivative[0, 3] = -1/4*(1+eta)

            derivative[1, 0] = -1/4*(1-xi)
            derivative[1, 1] = -1/4*(1+xi)
            derivative[1, 2] = 1/4*(1+xi)
            derivative[1, 3] = 1/4*(1-xi)
        if self.NPE == 8:
            pass
        return derivative

    def constitutive(self, i, j, k, l):

        c = (self.E / (2*(1 + self.nu))) * (globalstiffnesscalculation.delta(self, i, l) *\
        globalstiffnesscalculation.delta(self, j, k) + \
        globalstiffnesscalculation.delta(self, i, k) * \
        globalstiffnesscalculation.delta(self, j, l)) + ((self.E * self.nu)/(1- self.nu ** 2)) * \
        globalstiffnesscalculation.delta(self, i, j) * globalstiffnesscalculation.delta(self, k, l)
        return c

    def delta(self, a, b):
        if a == b :
            delta = 1
        else:
            delta = 0
        return delta
