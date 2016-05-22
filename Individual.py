# -*- coding: utf-8 -*-

import numpy as np

class Individual:
    """ 遺伝アルゴリズムで使用する個体クラス"""
    def __init__(self, length):
        self.chro = np.random.rand(length)*10-5
        self.fitness = 0

    # 個体の遺伝子型を取得する
    def getGeno(self):
        return self.chro

    # 遺伝子に載る数値をセットする
    def setValue(self, index, value):
        self.chro[index] = value

    # データに対して非線形変換を行う(シグモイド)
    def toNonLiner(self, data):
        # sigmoid
        return 1/(1+np.exp(-data))

    # 遺伝子から求めたい関数を生成して、dataを計算する
    def nextData(self, data):
        size = int((len(self.chro)-1)/3)
        w = self.chro[0: size]
        v = self.chro[size: 2*size]
        u = self.chro[2*size: 3*size]
        alph = self.chro[-1]
        data = self.toNonLiner(data)
        data = data*v+u
        alph = np.inner(data, w)+alph
        return alph

    # 教師データとdataの計算結果との差を計算する
    def calcValue(self, data, sample):
        self.value = abs(sample - self.nextData(data))

    # 教師データとdataの計算結果との差を取得する
    def getValue(self):
        return self.value

    # 個体の適応度を計算する
    # 教師データとdataの計算結果との差の総和を入力する必要がある
    def calcFitness(self, su):
        if su == 0:
            self.fitness = 0
            print("Sumation Equal Zero")
        else:
            self.fitness = 1-self.getValue()/su
            #print(self.fitness)

    # 個体の適応度を取得する
    def getFitness(self):
        return self.fitness

