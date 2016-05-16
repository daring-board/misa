# -*- coding: utf-8 -*-
import numpy as np

class SVR:

    err = 0.001
    eps = 0
    rate = 0.2

    def __init__(self, data_y, data_x):
        self.y = data_y
        self.x = data_x

    def gaussian(self, x):
        tmp = []
        for x_n in x:
            x_n = np.exp(-x_n/2)
            tmp.append(x_n)
        return np.array(tmp)

    def kernel(self, x1, x2):
        x = -np.linalg.norm(x1-x2)
        return np.exp(x/2)

    def getFunction(self, nalp, malp):
        alp = nalp-malp
        term1 = np.inner(alp, self.y)
        term2 = self.eps*np.sum(nalp, malp)
        term3 = 0
        for n in range(alp.size):
            xn = self.x[n]
            for m in range(alp.size):
                xm = self.x[m]
                term3 += alp[n]*alp[m]*self.kernel(xn, xm)
        return term1-term2-term3/2

    def getPartGrad(self, alp, n, flag):
        xn = self.x[n]
        if(flag):
            grad = self.y[n]-self.eps
            xn = self.x[n]
            for m in range(alp.size):
                xm = self.x[m]
                grad -= alp[m]*self.kernel(xn, xm)/2
        else:
            grad = -self.y[n]-self.eps
            xn = self.x[n]
            for m in range(alp.size):
                xm = self.x[m]
                grad += alp[m]*self.kernel(xn, xm)/2
        return grad

    def getGradient(self, nalp, malp):
        alp = nalp-malp
        grad_n = np.zeros(alp.size)
        grad_b = np.zeros(alp.size)
        for n in range(alp.size):
            grad_n[n] = self.getPartGrad(alp, n, True)
        for n in range(alp.size):
            grad_b[n] = self.getPartGrad(alp, n, False)
        return(grad_n, grad_b)

    def hillClimbing(self):
        nalp = np.random.rand(len(self.y))
        balp = np.random.rand(len(self.y))
        x = np.random.rand(2*len(self.y))
        norm_dx = 1
        while norm_dx > self.err:
            t_grad = self.getGradient(nalp, balp)
            step = self.rate
            #print("step:"+str(step))
            grad = np.r_[t_grad[0], t_grad[1]]
            d_x = step*grad
            x  = x + d_x
            norm_dx = np.linalg.norm(d_x)
            nalp += step*t_grad[0]
            balp += step*t_grad[1]
            #print(d_x)
            print(norm_dx)
        return (nalp, balp)

    def creatResult(self, alp, data):
        b = self.getPartGrad(alp[0], 0, True)
        y = 0
        for n in range(self.y.size):
            x2 = self.x[n]
            y += (alp[0][n]-alp[1][n])*self.kernel(data, x2)
        y += b
        return y

    def calcLoss(self, data_x, data_y):
        y = self.creatResult(self.result, data_x)
        loss = (y-data_y)*(y-data_y)
        return loss

    def update(self, data_x, data_y):
        x = []
        if self.calcLoss(data_x, data_y) > self.eps:
            ngrad = self.result[0]
            bgrad = self.result[1]
            d_x = self.rate*self.getGradient(ngrad, bgrad)
            x  = self.result + d_x
            self.resutl = np.array(x)
            #print(self.result)

    def executeLearn(self):
        alp1 = np.ones(2*len(self.y))
        self.result = self.hillClimbing()

    def getResult(self, data):
        return self.creatResult(self.result, data)
                
            
        
