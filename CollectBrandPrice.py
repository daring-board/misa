# -*- coding: utf-8 -*-

import csv
import jsm
import datetime
import numpy as np

class CollectBrandPrices:
    '''
    '''

    def __init__(self, codes):
        ''' constructor	'''
        self.brand_codes = codes
        self.close_value = []

    def collect(self, start_date, end_date):
        ''' jsm を用いて株価の収集を行います。
        テストモジュールのため./Initialize/brand_prices.csvを出力します。
        TODO: MonboDBに格納し、利用しやすい形に成型する
        '''
        f = open('close_data.csv', 'w')
        writer = csv.writer(f)
        q = jsm.Quotes()
        for code in self.brand_codes:
            print(code[0])
            try:
                data = q.get_historical_prices(code[0], jsm.DAILY, start_date, end_date)
                print(data)
                value_list =[]
                for d in data:
                    value_list.append(d.close)
                writer.writerow(value_list)
            except:
                print("exception occuer")
                pass
        f.close()

    def getClose(self):
        f = open('close_data.csv', 'r')
        reader = csv.reader(f)
        data = []
        for row in reader:
            data.append(row)
        print(data)
        return data

if __name__ == '__main__':
    f = open('brand_codes.csv', 'r')
    codes = csv.reader(f)
    cbp = CollectBrandPrices(codes)
    
    try:
        cbp.collect(datetime.date(2016, 5, 17), datetime.date(2016, 5, 19))
    except:
        pass
    f.close()
