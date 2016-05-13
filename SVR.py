# -*- coding: utf-8 -*-
import numpy as np

class SVR:

    err = 0.002
    eps = 0

    def __init__(self, data_y, data_x, rate):
        self.y = data_y
        self.x = data_x
        self.rate = rate

    def gaussian(self, x):
        tmp = []
        for x_n in x:
            x_n = np.exp(-x_n/2)
            tmp.append(x_n)
        return np.array(tmp)

    def kernel(self, x1, x2):
        x = -np.linalg.norm(x1-x2)
        return np.exp(x/2)

    def getPartGrad(self, alp, x, index):
        s = 0
        count = 0
        i = int(index/2) if(index%2==0) else int((index-1)/2)
        while count < len(self.y):
            x2 = self.x[count]
            s += (alp[2*count]-alp[2*count+1])*self.kernel(x, x2)
            count += 1
        result = self.y[i]-s/2-self.eps if(index%2==0) else -self.y[i]-self.eps+s/2
        return result
 
    def getGradient(self, alp):
        count = 0
        n_alp = []
        while count < len(self.y):
            x1 = self.x[count]
            n_alp.append(self.getPartGrad(alp, x1, 2*count))
            n_alp.append(self.getPartGrad(alp, x1, 2*count+1))
            count += 1
        return np.array(n_alp)

    def calcLs(self, step, alp, grad):
        funcL = self.getFunction(alp)
        diffL = 0
        sub = 0
        count = 0
        while count < len(self.y):
            ln = grad[2*count]-grad[2*count+1]
            kn = grad[2*count]+grad[2*count+1]
            sub += ln-self.eps*kn
            count += 1
        funcL += step*sub
        diffL += sub

        n = 0
        m = 0
        sub1 = 0
        sub2 = 0
        while n < len(self.y):
            ln = grad[2*n]-grad[2*n+1]
            an = alp[2*n]-alp[2*n+1]
            xn = self.x[n]
            while m < len(self.y):
                lm = grad[2*m]-grad[2*m+1]
                km = grad[2*m]+grad[2*m+1]
                am = alp[2*m]-alp[2*m+1]
                xm = self.x[m]
                sub1 = ln*lm*self.kernel(xn, xm)
                sub2 = ln*am+km*an*self.kernel(xn, xm)
                m += 1
            n += 1
        funcL -= step*step*sub1/2
        funcL -= step*sub2/2
        diffL -= (step*sub1+sub2/2)
        diff2L = -sub1
        return(funcL, diffL, diff2L)
    
    def newtonMethod(self, alf, grad):
        x = 1
        xb = 2
        while abs(x-xb) < self.eps:
            xb = x
            funcs = self.calcLs(xb, alp, grad)
            x = xb + funcs[1]/funcs[2]
        return x
        
    def exactSol(self, alp, grad):
        sub = 0
        count = 0
        while count < len(self.y):
            ln = grad[2*count]-grad[2*count+1]
            kn = grad[2*count]+grad[2*count+1]
            sub += ln-self.eps*kn
            count += 1
        b = sub

        n = 0
        m = 0
        sub1 = 0
        sub2 = 0
        while n < len(self.y):
            ln = grad[2*n]-grad[2*n+1]
            an = alp[2*n]-alp[2*n+1]
            xn = self.x[n]
            while m < len(self.y):
                lm = grad[2*m]-grad[2*m+1]
                km = grad[2*m]+grad[2*m+1]
                am = alp[2*m]-alp[2*m+1]
                xm = self.x[m]
                sub1 = ln*lm*self.kernel(xn, xm)
                sub2 = ln*am+km*an*self.kernel(xn, xm)
                m += 1
            n += 1
        a = sub1
        b += sub2/2
        print(str(a)+", "+str(b))
        x = 0.2 if(a < self.err) else -b/a
        return x

    def getFunction(self, alp):
        count = 0
        palp = np.zeros(int(len(alp)/2))
        malp = np.zeros(int(len(alp)/2))
        while count < len(alp)/2:
            palp[count] = alp[2*count]
            malp[count] = alp[2*count+1]
            count += 1
        term1 = np.inner(palp-malp, self.y)
        term2 = self.eps*np.sum(palp+malp)
        term3 = 0
        count1 = 0
        count2 = 0
        while count1 < len(self.y):
            x1 = self.x[count1]
            while count2 < len(self.y):
                x2 = self.x[count2]
                term3 += (palp[count1]-malp[count1])*(palp[count2]-malp[count2])*self.kernel(x1, x2)
                count2 += 1
            count1 += 1
        return term1-term2-term3/2

    def hillClimbing(self):
        x = np.random.rand(2*len(self.y))
        norm_dx = 100
        while norm_dx > self.err:
            grad = self.getGradient(x)
            step = self.exactSol(x, grad)
            #step = self.rate
            print("step:"+str(step))
            d_x = step*grad
            x  = x + d_x
            norm_dx = np.linalg.norm(d_x)
            #print(d_x)
            print(norm_dx)
        return x

    def creatResult(self, alp, data):
        b = self.getPartGrad(alp, data, 0)
        y = 0
        count = 0
        while count < len(self.y):
            x2 = self.x[count]
            y += (alp[2*count]-alp[2*count+1])*self.kernel(data, x2)
            count += 1
        y += b
        return y

    def calcLoss(self, data_x, data_y):
        y = self.creatResult(self.result, data_x)
        loss = (y-data_y)*(y-data_y)
        return loss

    def update(self, data_x, data_y):
        x = []
        if self.calcLoss(data_x, data_y) > self.eps:
            d_x = self.rate*self.getGradient(self.result)
            x  = self.result + d_x
            self.resutl = np.array(x)
            #print(self.result)

    def executeLearn(self):
        alp1 = np.ones(2*len(self.y))
        alp1 = self.getGradient(alp1)
        self.result = self.hillClimbing()

    def getResult(self, data):
        return self.creatResult(self.result, data)


        
