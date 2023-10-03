import numpy as np

def euler_solver(func, y0, t):
    y = np.zeros(shape=(len(t),len(y0)),dtype=np.float32)
    y[0,:] = y0
    for i in range(len(t)-1):
        y[i+1,:] = y[i,:] + (t[i+1]-t[i])*func(t[i],y[i,:])
    return y