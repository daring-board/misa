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
    #cbp.collect(datetime.date(2016, 5, 17), datetime.date(2016, 5, 20))
    predata = cbp.getClose()
    ldata = []
    real = 0
    #教師データ
    code = 0
    last = 20-17
    teach = 0
    counter = 0
    day = 0
    for pd in predata:
        for d in pd:
            if(day!=last):
                ldata.append(int(d))
            if (code == counter) and (day == last-1):
                teach = int(d)
            if(code == counter) and (day == last):
                real = int(d)
            day += 1
        counter += 1
    data = np.array(ldata)
    print(data)
    popSize = 800 # 遺伝アルゴリズムの個体数
    rga = ga.GeneticAlgorithm(popSize, data, teach)
    v_func = rga.execute(1000, 6)
    # 取得した表現行列から結果を計算
    result = rga.funcResult(v_func, data)
    print("teach Data = "+str(teach)+", result = "+str(result)+", \n real = "+str(real)+", error = "+str(abs(real-result)))
