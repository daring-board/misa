# -*- coding: utf-8 -*-
import numpy as np

class SVR:

    #ハイパーパラメータ
    err = 0.01  #許容誤差
    eps = 0     #モデル関数の定数
    rate = 0.2  #学習率(勾配降下法のステップ幅)

    """サポートベクトル回帰(SVR)"""
    def __init__(self, data_y, data_x):
        self.y = data_y
        self.x = data_x

    """ガウシアン 正規分布関数因子"""
    def gaussian(self, x):
        tmp = []
        for x_n in x:
            x_n = np.exp(-x_n/2)
            tmp.append(x_n)
        return np.array(tmp)

    """ ガウシアンカーネル """
    def kernel(self, x1, x2):
        x = -np.linalg.norm(x1-x2)
        return np.exp(x/2)

    """ サポートベクトル回帰のための目的関数"""
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

    """ 勾配ベクトルの各要素を計算する"""
    def getPartGrad(self, alp, n, flag):
        xn = self.x[n]
        if(flag):
            grad = self.y[n]-self.eps
            for m in range(alp.size):
                xm = self.x[m]
                grad -= alp[m]*self.kernel(xn, xm)/2
        else:
            grad = -self.y[n]-self.eps
            for m in range(alp.size):
                xm = self.x[m]
                grad += alp[m]*self.kernel(xn, xm)/2
        return grad

    """ 勾配ベクトルを計算する"""
    def getGradient(self, nalp, malp):
        alp = nalp-malp
        grad_n = np.zeros(alp.size)
        grad_b = np.zeros(alp.size)
        for n in range(alp.size):
            grad_n[n] = self.getPartGrad(alp, n, True)
        for n in range(alp.size):
            grad_b[n] = self.getPartGrad(alp, n, False)
        return(grad_n, grad_b)

    """ 目的関数を最大化するような変数nalpとbalpを求める
　　　　最適値を求める必要はない"""
    def hillClimbing(self):
        nalp = np.zeros(len(self.y))
        balp = np.zeros(len(self.y))
        x = np.random.rand(2*len(self.y))
        norm_dx = 1
        while norm_dx > self.err:
            grad = self.getGradient(nalp, balp)
            step = self.rate
            norm_dx = np.linalg.norm(np.r_[step*grad[0], step*grad[1]])
            nalp += step*grad[0]
            balp += step*grad[1]
            #print(d_x)
            #print(norm_dx)
        return (nalp, balp)

    """ 最適値を取るような変数alpを用いて、
　　　 dataから推定値を計算する"""
    def creatResult(self, alp, data):
        b = 0
        y = 0
        for n in range(self.y.size):
            b += self.getPartGrad(alp[0], n, True)
            x2 = self.x[n]
            y += (alp[0][n]-alp[1][n])*self.kernel(data, x2)/2
        y += b/self.y.size
        return y

    """ オンライン学習のための損失関数 """
    def calcLoss(self, data_x, data_y):
        y = self.creatResult(self.result, data_x)
        loss = (y-data_y)*(y-data_y)
        return loss

    """ オンライン学習を行う """
    def update(self, data_x, data_y):
        if self.calcLoss(data_x, data_y) > self.eps:
            ngrad = self.result[0]
            bgrad = self.result[1]
            grad = self.getGradient(ngrad, bgrad)
            ngrad += self.rate*grad[0]
            bgrad += self.rate*grad[1]
            self.result = (ngrad, bgrad)

    """ サポートベクトル回帰で関数を学習させる """
    def executeLearn(self):
        self.result = self.hillClimbing()

    """ 入力したdataから推定される値を返却する"""
    def getResult(self, data):
        return self.creatResult(self.result, data)
                
            
        
