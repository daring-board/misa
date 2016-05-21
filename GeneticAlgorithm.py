# -*- coding: utf-8 -*-

import Individual as ind
import numpy as np
import numpy.random as random

class GeneticAlgorithm:
    """ 遺伝アルゴリズム(MGG)"""
    def __init__(self, size, data, sample):
        self.inds = []
        self.populationSize = size
        self.chroSize = 3*len(data)+1
        self.rate = 0.2
        self.data = data
        self.teach = sample
        self.sFit = 0

    # test    
    def getName(self):
        return self.name

    # test
    def setName(self, name):
        self.name = name

    # 個体をランダムに作成する
    def createPopulation(self):
        count = 0
        while count < self.populationSize:
            count += 1
            i = ind.Individual(self.chroSize)
            self.inds.append(i)

    # 全個体の遺伝子を表示する
    def printIndividual(self):
        for i in self.inds:
            print(i.getGeno())

    # グループ内の全個体の遺伝子を表示する
    def printIndividual(self, group):
        for i in group:
            print(i.getGeno())

    # 子を作成する親個体を選択する
    def selectTwoIndividual(self):
        select = []
        counter = 0
        while counter < 2:
            index = random.randint(len(self.inds))
            select.append(self.inds[index])
            del self.inds[index]
            counter += 1
        return select

    # ◆交叉
    # 選択した親個体から子個体1体を作成する(BLX-0.2)
    def crossover(self, parent):
        child = ind.Individual(self.chroSize)
        counter = 0
        p0 = parent[0].getGeno()
        p1 = parent[1].getGeno()
        while counter < self.chroSize:
            dist = abs(p0[counter] - p1[counter])
            tup = (p0[counter], p1[counter])
            minimal = min(tup)
            maximal= max(tup)
            minimal -= dist*self.rate
            maximal += dist*self.rate
            child.setValue(counter, random.rand()*(maximal-minimal)+minimal)
            counter += 1
        return child

    # グループ内で適応度が最大の個体を選択する
    def getBest(self, family):
        index = 0
        count = 0
        maxFitness = 0
        best = ind.Individual(self.chroSize)
        while count < len(family):
            ch = family[count]
            ch.calcFitness(self.sFit)
            if maxFitness < ch.getFitness():
                maxFitness = ch.getFitness()
                best = ch
                index = count
            count += 1
        del family[index]
        return best

    # グループ内の個体を適応度に比例した確率で1体選択する
    def getByRoulette(self, family):
        prob = []
        for ch in family:
            prob.append(ch.getFitness())
        individual = random.choice(family, 1, prob)
        return individual[0]

    # ◆選択
    # 次の世代に残す個体を親とその子の家族の中から2体選択する
    def selection(self, children, parent):
        counter = 0
        family = []
        select = []
        for ch in parent: family.append(ch)
        for ch in children: family.append(ch)
        for ch in family:
            ch.calcValue(self.data, self.teach)
            self.sFit += ch.getValue()
        select.append(self.getBest(family))
        select.append(self.getByRoulette(family))
        self.sFit = 0
        return select

    # 遺伝アルゴリズムの実行　世代交代回数、子個体の数を指定
    def execute(self, repeat, childrenSize):
        self.createPopulation()
        counter = 0
        while counter < repeat:
            parent = self.selectTwoIndividual()
            rep = 0
            children = []
            while rep < childrenSize:
                children.append( self.crossover(parent) )
                rep += 1
            select = self.selection(children, parent)
            self.inds.append(select[0])
            self.inds.append(select[1])
            counter += 1
            #print(str(counter)+":"+str(len(self.inds)))
        for ch in self.inds:
            ch.calcValue(self.data, self.teach)
            self.sFit += ch.getValue()
        best = self.getBest(self.inds)
        print(best.getGeno())
        return best

    # 結果の計算　個体の表現関数にdataを入力して計算
    def funcResult(self, result, data):
        return result.nextData(data)
    
