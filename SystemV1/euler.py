import numpy as np
from tqdm import tqdm

def euler_solver(func, y0, t,name_by_index,debug=False):
    y = np.zeros(shape=(len(t),len(y0)),dtype=np.float32)
    y[0,:] = y0
    for i in range(len(t)-1):
        dYdt = func(t[i],y[i,:])
        y[i+1,:] = y[i,:] + (t[i+1]-t[i])*dYdt
        if debug:
            where_bad_i = np.argwhere(np.abs(dYdt)> 10**4).flatten()
            if len( where_bad_i ) !=0:
                print('t\t{}'.format(t[i]))
                for j in where_bad_i:
                    print('d{}dt(t)\t{}'.format(name_by_index[j],dYdt[j]))
                    print('{}(t)\t{}'.format(name_by_index[j],y[i,j]))
                break
    return y