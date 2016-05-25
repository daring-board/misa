# -*- coding: utf-8 -*-
import numpy as np
import SVR as svm
import csv
import CollectBrandPrice as price
import datetime

if __name__ == "__main__": #EntryPoint
    f = open('brand_codes.csv', 'r')
    codes = csv.reader(f)
    cbp = price.CollectBrandPrices(codes)
    #cbp.collect(datetime.date(2016, 4, 1), datetime.date(2016, 5, 20))
    value = cbp.getClose()
    data_set = []
    for n in range(len(value[0])): data_set.append([])
    for d in value:
        #print(len(d))
        if(len(d) != 32): continue
        for day in range(len(d)):
            data_set[day].append(int(d[day]))


    code = 4
    #教師データの作成(4/2～5/19)
    yd = []
    for data in data_set:
        yd.append(data[code])
    del yd[code]
    yd.pop()
    print("教師データ：")
    print(str(yd))
    #print(len(yd))
    #入力データの作成(4/1～5/18)
    xd_set = []
    for x in data_set:
        xd_set.append(x)
    xd_set.pop()
    xd_set.pop()
    print("学習データ：")
    for xd in xd_set:
        print(xd)
    #print(len(xd_set))

    x_set = np.array(xd_set)
    y = np.array(yd)
    svr = svm.SVR( y, x_set)
    svr.executeLearn()

    print(data_set[len(data_set)-2])
    y = svr.getResult(data_set[len(data_set)-2])
    print("result = " + str(y) + ", real = "+str(data_set[len(data_set)-1][code])+", error = " + str(data_set[len(data_set)-1][code] - y))
    #print(x_set[28])
    #y = svr.getResult(x_set[28])
    #print("result = " + str(y) + ", real = "+str(x_set[29][code])+", error = " + str(x_set[29][code] - y))
