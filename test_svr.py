# -*- coding: utf-8 -*-
import numpy as np
import SVR as svm

if __name__ == "__main__": #EntryPoint
    size = 25
    count = 0
    set_tmp = []
    np.random.seed(0)
    while count < size:
        x = [count, np.pi/2]
        set_tmp.append(x)
        count += 1
    set_x = np.array(set_tmp)
    count = 0
    y_t = []
    while count < size:
        y_t.append(np.sin(set_x[count][0]*set_x[count][1]))
        count += 1
    y = np.array(y_t)
    for x in set_x:
        print(x)
    print("sample :"+str(y))
    
    svr = svm.SVR( y,set_x)
    svr.executeLearn()

    count = 0
    while count < 10:
        x = [count, np.pi/2]
        print("Input:")
        print(x)
        y = svr.getResult(x)
        print("Result:"+str(y))
        svr.update(x,0)
        count += 1
