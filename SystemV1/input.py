import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import ode
from myo_supportfs import *
import os
import scipy.integrate as integrate
from numba import jit
# from numba.experimental import jitclass
# from numba import int32, float32
# from typing import List

from typing import Any
from tqdm import tqdm
from tslearn import generators

def plot_float_distribution(data,fig_size=(4,3),title=''):
    fig, ax = plt.subplots()
    fig.set_size_inches(fig_size[0],fig_size[1])
    x = []
    for i in range(len(data)):
        if np.isnan(data[i]):
            continue
        else:
            x.append(data[i])
    u_vs = np.unique(x)

    if len(x) == 0:
        ax.set_title(title +' is empty data')
    elif len(u_vs)==1:
        ax.set_title(title + ' all data is repeated with value: {}'.format(u_vs[0]))
    else:
        x = np.asarray(x)
        q25, q75 = np.percentile(x, [25, 75])
        bins = 0
        if q25==q75:
            bins = np.minimum(100,len(u_vs))
        else:
            bin_width = 2 * (q75 - q25) * len(x) ** (-1 / 3)
            bins = np.minimum(100, round((np.max(x) - np.min(x)) / bin_width))
        nan_rate = np.sum(np.isnan(data))/len(data)
        ax.set_title(title+'. n of unique values {}'.format(len(u_vs)))
        ax.set_xlabel('nan rate {}'.format(nan_rate))
        density,bins = np.histogram(x,bins=bins,density=True)
        unity_density = density / density.sum()
        widths = bins[:-1] - bins[1:]
        ax.bar(bins[1:], unity_density,width=widths)

    return fig,ax

@jit(nopython=True)
def Heviside(x:float) -> float:
    if x < 0.0:
        return 0.0
    return 1.0
@jit(nopython=True)
def LeftHeviside(x:float) -> float:
    if x <= 0.0:
        return 0.0
    return 1.0

@jit(nopython=True)
def RightHeviside(x:float) -> float:
    if x < 0.0:
        return 0.0
    return 1.0


class J_ch:
    t1:                     float
    t2:                     float
    delta_t:                float
    tau:                    float
    T:                      float
    rho:                    float
    alpha:                  float
    volume:                 float
    start_absorbtion:       float
    stop_absorbtion:        float
    mass_before_absorbtion: float
    last_J:                 float
    last_delta_J:           float
    is_mass_dont_used_up:   int

    def __init__(self, t1:float, t2:float,delta_t:float, tau:float, T:float,rho:float,alpha:float,volume:float) -> None:
        self.t1 = t1
        self.t2 = t2
        self.delta_t = delta_t
        self.tau = tau
        self.T = T
        self.rho = rho
        self.alpha = alpha
        self.volume = volume
        self.start_absorbtion = t2 + tau
        self.stop_absorbtion = t2 + tau + T 
        self.mass_before_absorbtion = (1.0/volume)*alpha*rho*(t2-t1)
        self.last_delta_J = 0.0
        self.last_J = self.mass_before_absorbtion
        self.is_mass_dont_used_up = 1
    def step(self, t: float, velocity: float) -> None:
        if (t < self.start_absorbtion or t > self.stop_absorbtion) or not self.is_mass_dont_used_up:
            return
        else:
            self.last_delta_J = -velocity*self.delta_t
            self.last_J = np.maximum(self.last_J + self.last_delta_J, 0.0)
            if self.last_J == 0.0:
                self.last_delta_J = 0.0
                self.is_mass_dont_used_up = 0


    def get_J(self, t:float):
        # return self.last_J * Heviside(t - self.start_absorbtion) * Heviside(self.stop_absorbtion - t)
        if t < self.t1:
            return 0.0
        elif t >= self.t1 and t <= self.t2:
            return self.rho * (t - self.t1) * self.alpha/ self.volume
        elif t > self.t2 and t < self.start_absorbtion:
            return self.mass_before_absorbtion
        else:
            return self.last_J * int(t>=self.start_absorbtion and t <= self.stop_absorbtion)
    def get_dJdt(self, t:float):
        # return self.last_delta_J/self.delta_t * Heviside(t - self.start_absorbtion) * Heviside(self.stop_absorbtion - t)
        return self.last_delta_J/self.delta_t * int(t>=self.start_absorbtion and t <= self.stop_absorbtion)
    
class J_sum:
    J_arr: List[J_ch]
    V_total: float
    def __init__(self, V_total) -> None:
        self.V_total = V_total
        self.J_arr = []
    def add_J_ch(self, t1:float, t2:float, delta_t:float, tau:float, T:float,rho:float,alpha:float, volume:float):
        J_ = J_ch(t1,t2,delta_t,tau,T,rho,alpha,volume)
        self.J_arr.append(J_)
    def get_dJdt(self,t:float)->float:
        s_ = 0.0
        for i in range(len(self.J_arr)):
            s_ += self.J_arr[i].get_dJdt(t)
        return s_
    
    def get_J(self,t:float)->float:
        s_ = 0.0
        # arr_of_values = np.zeros(shape=(len(self.J_arr),))
        for i in range(len(self.J_arr)):
            # arr_of_values[i] = self.J_arr[i].get_J(t) 
            s_ += self.J_arr[i].get_J(t)
        return s_

    def get_velocity(self, t: float) -> float:
        num_ = 0.0
        for i in range(len(self.J_arr)):
            J_ = self.J_arr[i]
            # num_ += Heviside(t - J_.start_absorbtion) * Heviside(J_.stop_absorbtion - t)*J_.is_mass_dont_used_up
            num_ += int(t>=J_.start_absorbtion and t <= J_.stop_absorbtion)*J_.is_mass_dont_used_up
        num_ = int(num_)
        if num_ == 0.0:
            return 0.0
        else:
            return self.V_total/num_
        
    def step(self,t:float):
        V = self.get_velocity(t)
        for i in range(len(self.J_arr)):
            self.J_arr[i].step(t, velocity=V)


def itegrate_func(func, time_grid):
    out_ = np.zeros(shape=(len(time_grid),))
    sum_= 0.0
    for i in range(1,len(time_grid)):
        tau_i = time_grid[i]-time_grid[i-1]
        mid_t = 0.5*(time_grid[i]+time_grid[i-1])
        mid_point = func(mid_t)
        sum_ += tau_i*mid_point
        out_[i] = sum_
    return out_
def itegrate_arr(arr_, time_grid):
    out_ = np.zeros(shape=(len(time_grid),))
    sum_= 0.0
    for i in range(1,len(time_grid)):
        tau_i = time_grid[i]-time_grid[i-1]
        right_point = arr_[i]
        sum_ += tau_i*right_point
        out_[i] = sum_
    return out_

def simple_plot_many_y(x,y_vec,title='', labels=[],fig_size = (4,4),scale= 'linear'):
    fig, ax = plt.subplots()
    fig.set_size_inches(fig_size)
    for i in range(len(y_vec)):
        y_ = y_vec[i]
        x_ = x
        if len(labels) == len(y_vec):
            ax.plot(x_,y_,label = labels[i])
        else:
            ax.plot(x_,y_)
    ax.legend()
    ax.set_title(title)
    ax.set_yscale(scale)
    ax.grid()
    return fig,ax

@jit(nopython=True)
def simple_search_position_on_grid(grid:np.array,value:float)->int:
    # return segment index to which balue belongs
    N = len(grid)
    for i in range(N-1):
        if value >= grid[i] and value <= grid[i+1]:
            return i

class func_on_linear_grid:
    values: np.array
    tau: float
    t_0:float
    t_end:float
    def __init__(self, tau,t_0,t_end,values):
        self.tau = tau
        self.t_0 = t_0
        self.t_end = t_end
        self.values  = values
    def __call__(self, t:float)->float:
        if not (t < self.t_0 or t > self.t_end):
            i = int(np.rint((t-self.t_0)/self.tau))
            return self.values[i]
        else:
            return 0.0

def HeartRate_gen(tau, time_grid,hlow=60,hhigh=120):
    N  = len(time_grid)
    y_=  generators.random_walks(n_ts=1,sz=N).flatten()
    a_ = np.min(y_)
    b_ = np.max(y_)
    y_ = (y_-a_)/(b_-a_)
    c_ = hlow
    d_ = hhigh
    y_ *= (d_-c_)
    y_ += c_
    return func_on_linear_grid(tau,time_grid[0],time_grid[-1],y_) 

class ClosestValueFinder:
    array = None
    # [deviation in minutes]
    def __init__(self, array,tau, max_deviation=2):
        self.tau = tau
        self.array = array
        self.prev_index = None
        self.max_deviation = max_deviation  # Максимальное отклонение для использования окрестности

    def find_closest_pos(self, t):
        if self.prev_index is not None:
            left = max(0, self.prev_index - int(self.max_deviation/self.tau))
            right = min(len(self.array) - 1, self.prev_index + int(self.max_deviation/self.tau))
        else:
            left = 0
            right = len(self.array) - 1

        # closest = None
        closest_pos = None
        min_diff = float('inf')

        # print(left,right)
        for i in range(left, right + 1):
            diff = abs(self.array[i] - t)
            if diff < min_diff:
                # closest = self.array[i]
                closest_pos = i
                min_diff = diff
                self.prev_index = i

        if min_diff > self.max_deviation:
            print('bad find')
            # Если отклонение слишком большое, выполнить полный бинарный поиск
            left = 0
            right = len(self.array) - 1
            while left <= right:
                mid = left + (right - left) // 2
                if self.array[mid] == t:
                    return t
                elif self.array[mid] < t:
                    left = mid + 1
                else:
                    right = mid - 1

            # Найдено ближайшее значение с помощью полного бинарного поиска
            if abs(self.array[left] - t) < abs(self.array[right] - t):
                closest_pos = left 
            else:
                closest_pos = right
            # return self.array[left] if abs(self.array[left] - t) < abs(self.array[right] - t) else self.array[right]
        return closest_pos
    

def AUC_x_vec_y_vec(x,y,i1,i2):
    s = 0.0
    for j in range(i1, i2):
        s += 0.5*(y[j]+y[j+1])*(x[j+1]-x[j])
    return s
