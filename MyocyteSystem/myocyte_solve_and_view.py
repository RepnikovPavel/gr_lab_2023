import torch
import myocyte_de_config
import config as config
import numpy as np
from myocyte_gen_des import F_vec
from scipy.integrate import odeint
from Plotting.myplt import *
from aml.time_mesuarment import timeit
from pprint import pprint as Print
from matplotlib import collections as matcoll
from numba import jit
from ParsingSystem.parse_and_build import TableGenerator

@jit(nopython=True, cache=True)
def Loss(J_at_ith_time:np.array)->float:
    J_abs = np.absolute(J_at_ith_time)
    argmax_ = np.argmax(J_abs,axis=1)
    loss_ = 0.0
    for i in range(len(J_abs)):
        j = argmax_[i]
        sum_= 0.0
        for k in range(len(J_abs[i])):
            if k != j:
                sum_+=  np.absolute(J_abs[i][k]-J_abs[i][j])
        sum_ = sum_ * (1/(len(J_abs[i])-1))
        loss_ += sum_
    loss_ = loss_ * (1/len(J_abs))
    return loss_

@jit(nopython=True, cache=True)
def LossMetric(J_at_ith_time:np.array)->float:
    # доля неработающих параметров
    J_abs = np.absolute(J_at_ith_time)
    LossMetric = 0
    all_ = 0
    for i in range(len(J_abs)):
        for k in range(len(J_abs[i])):
            all_ +=1
            if J_abs[i][k] ==0.0:
                LossMetric +=1
    return LossMetric/all_*100.0

def get_params_segments(params:np.array,from_index_of_param_to_param_name):
    o_ = np.zeros(shape=(len(params),2))
    percent  = 0.4
    for i in range(len(o_)):
        param_name_ = from_index_of_param_to_param_name[i]
        if param_name_ == r'$\tau_{carb}$' or param_name_ == r'$\tau_{fat}$' or param_name_==r'$\tau_{prot}$':
            o_[i][0] = params[i]
            o_[i][1] = params[i]
        if params[i] < 0.0:
            o_[i][0]= params[i]+percent*params[i]
            o_[i][1]= params[i]-percent*params[i]
        else:
            o_[i][0]= params[i]-percent*params[i]
            o_[i][1]= params[i]+percent*params[i]
    return o_


@jit(nopython=True, cache=True)
def gen_params(params_segments:np.array):
    o_ = np.zeros(shape=(len(params_segments),))
    for i in range(len(o_)):
        o_[i] = np.random.uniform(low=params_segments[i][0],high=params_segments[i][1])
    return o_

def grad(params_vec,time_grid, start_point):
    dth = 10**(-6)
    params_for_grad  = np.zeros(shape=(2*len(params_vec),len(params_vec)))
    k_=0
    for i in range(0,len(params_for_grad),2):
        params_for_grad[i] = params_vec
        params_for_grad[i+1] = params_vec
        params_for_grad[i][k_] = params_vec[k_]+dth  
        params_for_grad[i+1][k_] = params_vec[k_]-dth
        k_+=1
    # J__ = \frac{DY}{D \Theta} \at TimeGrid[i]
    J_ = np.zeros(shape=(len(time_grid),len(params_vec),len(start_point)))
    k_ = 0
    for i in range(0,len(params_for_grad),2):
        # print('{}/{}'.format(k_,len(params_for_grad)/2))
        DYDthi= (1/(2*dth))*(odeint(func=F_vec, y0=start_point, t=time_grid, args=(params_for_grad[i],), full_output=False) - odeint(func=F_vec, y0=start_point, t=time_grid, args=(params_for_grad[i+1],), full_output=False))
        # DYDthi[i of time_grid,:] # (33,)
        J_[:,k_,:] = DYDthi 
        k_ += 1
    J__ = np.transpose(J_,(0,2,1))
    # J = np.zeros(shape=(len(time_grid),len(start_point),len(params_vec)))
    # for i in range(len(time_grid)):
        # matrix_  = J_[i]

        # J[i] = 
    return J__
    
def get_mvp_params(N_of_simulations, segments_, start_point,params_vec, time_grid, to_new_y_name, from_index_of_param_to_param_name):
    N_ = N_of_simulations
    best_params_ = params_vec
    J0 = grad(params_vec,time_grid,start_point)[-1]
    loss_best = Loss(J0)
    best_metric = LossMetric(J0)
    x_loss_vec = [0]
    y_loss_vec = [loss_best]
    y_metric_vec = [best_metric]

    for i in range(1,N_+1):
        print('{}/{} loss {} metric {}'.format(i,N_,loss_best,best_metric))
        params_ = gen_params(segments_)
        Ji = grad(params_,time_grid,start_point)[-1]
        loss_new_=  Loss(Ji)
        if loss_new_ < loss_best:
            loss_best = loss_new_
            best_params_ = params_
            best_metric = LossMetric(Ji)
            x_loss_vec.append(i)
            y_loss_vec.append(loss_best)
            y_metric_vec.append(best_metric)

    fig,axs = plt.subplots(nrows=1,ncols=2)
    fig.set_size_inches(16, 9)
    axs[0].plot(x_loss_vec,y_loss_vec)
    axs[1].plot(x_loss_vec,y_metric_vec)
    axs[0].set_title('$loss$')
    axs[0].set_xlabel('generation index')
    axs[1].set_xlabel('generation index')
    axs[1].set_title('$metric$')
    new_values_ = {}
    for i in range(len(best_params_)):
        new_values_.update({from_index_of_param_to_param_name[i]:best_params_[i]})
    Print(new_values_)

    tau = 0.01
    t_0 = 0.0
    t_end = 2*1440.0
    params_vec = best_params_
    time_grid = np.linspace(start=t_0, stop=t_end, num=N)
    # solutions = odeint(func=F_vec, y0=start_point, t=time_grid, args=(params_vec,), full_output=False)
    
    J = grad(params_vec,time_grid,start_point)
    time_index = int((t_oprim-t_0)/tau)
    J_at_ith_time = J[time_index]
    mvp_params_indices = np.argsort(np.absolute(J_at_ith_time))
    mvp_params = {}
    mvp_values_of_params = {}

    # criterion(J_at_ith_time)
    list_of_dead_params = []
    for i in range(len(mvp_params_indices)):
        desc = list(reversed(list(mvp_params_indices[i])))
        mvp_names = []
        mvp_values = []
        for ind in desc:
            if np.absolute(J_at_ith_time[i][ind]) == 0.0:
                list_of_dead_params.append(from_index_of_param_to_param_name[ind])
            else: 
                mvp_names.append(from_index_of_param_to_param_name[ind])
                mvp_values.append(np.absolute(J_at_ith_time[i][ind]))
        y_name_ = get_y_name_by_index_of_solution(index_of_solution = i,
                                                    from_new_to_old_y_names_dict=to_new_y_name)
        
        # lines = []
        # for i in range(len(mvp_values)):
        #     pair=[(i,0), (i, mvp_values[i])]
        #     lines.append(pair)

        # linecoll = matcoll.LineCollection(lines)
        # fig1,ax1 = plt.subplots()
        # ax1.add_collection(linecoll)
        # ax1.set_xticks(np.arange(start=0,stop=len(desc),dtype=np.intc),mvp_names,rotation=45)
        # ax1.scatter(np.arange(start=0,stop=len(desc)),mvp_values)
        # ax1.set_title('$'+y_name_+'$')
        # ax1.set_yscale('log')
        # plt.show()
        mvp_values_of_params.update({y_name_:mvp_values})
        mvp_params.update({y_name_:mvp_names})

        list_of_dead_params = list(set(list_of_dead_params))


    latex_equations = torch.load(config.latex_eq_path)
    table_generator = TableGenerator()
    rows = []
    dead_params_str_ = ''
    for i in range(len(list_of_dead_params)-1):
        p_ = list_of_dead_params[i]
        p_= p_[:-1]
        p_ = p_[1:]
        p_ = r'\('+ p_ + r'\)'
        dead_params_str_ +=  p_+','
    p_ = list_of_dead_params[-1]
    p_= p_[:-1]
    p_ = p_[1:]
    p_ = r'\('+ p_ + r'\)'
    dead_params_str_ += p_
    rows.append(['dead parameters at t={} min'.format(t_oprim), dead_params_str_])
    for solution_name in latex_equations.keys():
        eq_str = latex_equations[solution_name]
        ranked_params = mvp_params[solution_name]
        params_str_ = ''
        for i in range(len(ranked_params)-1):
            p_ = ranked_params[i]
            p_= p_[:-1]
            p_ = p_[1:]
            p_ = r'\('+ p_ + r'\)'
            params_str_ +=  p_+','
        p_ = ranked_params[-1]
        p_= p_[:-1]
        p_ = p_[1:]
        p_ = r'\('+ p_ + r'\)'
        params_str_ += p_
        rows.append([r'\('+ r'\frac{d}{dt}' + solution_name + '=' +eq_str+r'\)',params_str_])
    table_generator.start_table()
    table_generator.insert_rows(rows)
    table_generator.end_table()
    table_generator.render(path_to_rendered_html=os.path.join(myocyte_de_config.write_params_values_to,'params_values.html'))

    # fig = init_figure()
    # fig = plot_solutions(fig, solutions, time_grid, to_new_y_name)
    # FcarbTimeSeries = np.asarray([myocyte_de_config.F_carb(time_grid[i]) for i in range(len(time_grid))])
    # FfatTimeSeries = np.asarray([myocyte_de_config.F_fat(time_grid[i]) for i in range(len(time_grid))])
    # FprotTimeSeries = np.asarray([myocyte_de_config.F_prot(time_grid[i]) for i in range(len(time_grid))])
    # add_line_to_fig(fig, time_grid,FprotTimeSeries,'F_{prot}')
    # add_line_to_fig(fig, time_grid,FfatTimeSeries,'F_{fat}')
    # add_line_to_fig(fig, time_grid,FcarbTimeSeries,'F_{carb}')

    # fig.show()
    # save_fig_to_html(fig,path=config.myo_path_to_html,filename='solution.html')
    # plt.show()

    plt.show()    
    return best_params_,list_of_dead_params, mvp_values_of_params, mvp_params



if __name__ == '__main__':
    to_new_var_name = torch.load(config.myocyte_map_from_old_param_name_to_new_name_dict_filename)
    params_vec,from_index_of_param_to_param_name = make_params_vec_from_params_dict(source_params_values=myocyte_de_config.params_values,
                                                  from_source_param_name_to_new=to_new_var_name)
    to_new_y_name = torch.load(config.myocyte_map_from_old_y_name_to_new_name_dict_filename)

    start_point = get_start_point_values(source_y_dict_with_start_point=myocyte_de_config.start_point,
                                         from_source_y_name_to_new=to_new_y_name)
    # y_vec_len = start_point.shape[0]
    # start_point = np.random.uniform(low=0.5, high=2.5, size=y_vec_len)


    tau = 0.01
    t_0 = 0.0
    t_oprim = 2.0

    N = int((t_oprim-t_0)/tau)+1
    print('grid size {}'.format(N))
    time_grid = np.linspace(start=t_0, stop=t_oprim, num=N)


    segments_ = get_params_segments(params_vec,from_index_of_param_to_param_name)
    get_mvp_params(N_of_simulations=100,
                   segments_=segments_,
                   start_point=start_point,
                   params_vec=params_vec,
                   time_grid=time_grid,
                   to_new_y_name=to_new_y_name,
                   from_index_of_param_to_param_name=from_index_of_param_to_param_name)