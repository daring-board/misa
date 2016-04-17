# -*- coding: utf-8 -*-
#inital module
import datetime
import sys

print("モジュールのロード")


def test():
    print("関数：testを呼び出しました")

def getToday():
    today = datetime.date.today()
    value = (today.year, today.month, today.day)
    return(value)


if __name__ == "__main__": #EntryPoint
    print("python-izm")
#   print("パイソンイズム")
    test()
    test_str = "test1\ntest2"
    print(test_str)
    test_str = 'python'
    test_str += ("-"+'izm')
    print(test_str)
    test_int = 100
    print(str(test_int)+"円")
    print(str(test_int)+"円".replace("円", "個"))
    print(test_str.split("-"))
    test_str = "1234"
    print(test_str.rjust(10))
    print(test_str.rjust(10, "0"))
    print(test_str.zfill(10))
    print(test_str.rjust(2))
    print(test_str.rjust(2, "0"))
    print(test_str.zfill(2))
    print(datetime.date.today())
    print(datetime.datetime.today())
    test_tuple = getToday()
    print(test_tuple)
    print(str(test_tuple[0])+"."+str(test_tuple[1])+"."+str(test_tuple[2]))
    param = sys.argv
    print(param)
    for i in range(10):
        print(i,end="")
    counter = 0
    # Whileで終了判定に変数を指定する場合は先に定義しておく必要がある
    while counter < 10:
        counter += 1
        print(counter,end="")
    for j in range(-5,5):
        print(j,end="")
    print()
    function = lambda num1,num2: num1+num2
    print(function(5,5))

