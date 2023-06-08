import numpy as np
import matplotlib.pyplot as plt
import math

class Elemen_stiffnesscalculation:
    def __init__(self, x, GPE, NPE, PD, E, nu):
        self.x = x
        self.GPE = GPE
        self.NPE = NPE
        self.PD = PD
        self.E = E
        self.nu = nu

    def elementstiffness(self):
        NPE = np.size(self.x, 0)
        PD = np.size(self.x, 1)
        kelement = np.zeros([NPE*PD,NPE*PD])
        trans_x = self.x.T
        for i in range(1, NPE+1):
            for j in range(1, NPE+1):
                k_small = np.zeros([PD, PD])
                for gp in range(1, self.GPE+1):
                    Jacobian = np.zeros([PD, PD])
                    grad_integral = np.zeros([PD, NPE])
                    (xi, eta, alpha) = Elemen_stiffnesscalculation.Gausspoint(self, gp)
                    derivative = Elemen_stiffnesscalculation.grad_shapefunction(self, xi, eta)
                    Jacobian = trans_x @ derivative.T
                    grad_integral = np.linalg.inv(Jacobian).T @ derivative
                    for a in range(1,PD+1):
                        for c in range(1,PD+1):
                            for b in range(1,PD+1):
                                for d in range(1,PD+1):
                                    k_small[a-1, c-1] =  k_small[a-1, c-1] + (grad_integral[b-1, i-1] * \
                                    Elemen_stiffnesscalculation.constitutive(self, a, b, c, d) * grad_integral[d-1, j-1]) * \
                                    (np.linalg.det(Jacobian) * alpha)
                    kelement[((i-1)*PD+1)-1 : i*PD, ((j-1)*PD+1)-1: j*PD] = k_small
        return kelement


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
        c = (self.E/(2*(1+self.nu))) * (Elemen_stiffnesscalculation.delta(self, i, l) * Elemen_stiffnesscalculation.delta(self, j, k) + \
        Elemen_stiffnesscalculation.delta(self, i, k)*Elemen_stiffnesscalculation.delta(self, j, l)) + ((self.E*self.nu)/(1-self.nu**2)) * \
        Elemen_stiffnesscalculation.delta(self, i, j) * Elemen_stiffnesscalculation.delta(self, k, l)
        return c

    def delta(self, a, b):
        if a == b :
            delta = 1
        else:
            delta = 0
        return delta