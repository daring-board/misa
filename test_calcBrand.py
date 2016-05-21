# -*- coding: utf-8 -*-

import GeneticAlgorithm as ga
import numpy as np
import csv
import datetime
import CollectBrandPrice as price

# テストモジュール
if __name__ == "__main__":

    f = open('brand_codes.csv', 'r')
    codes = csv.reader(f)
    cbp = price.CollectBrandPrices(codes)
    cbp.collect(datetime.date(2016, 5, 17), datetime.date(2016, 5, 19))
    predata = cbp.getClose()
    ldata = []
    for pd in predata:
        for d in pd:
           ldata.append(d)
    data = np.array(ldata)
    print(data)
    teach = 200 # 教師データ
    popSize = 20 # 遺伝アルゴリズムの個体数
    rga = ga.GeneticAlgorithm(popSize, data, teach)
    v_func = rga.execute(500, 6)
    # 取得した表現行列から結果を計算
    result = rga.funcResult(v_func, data)
    print("teach Data = "+str(teach)+", result = "+str(result)+", error = "+str(abs(teach-result)))
