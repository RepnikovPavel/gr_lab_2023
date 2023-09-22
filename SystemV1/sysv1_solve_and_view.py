import torch
import sysv1_de_config
import config as config
import numpy as np
from gen_des import F_vec
from scipy.integrate import odeint
from Plotting.myplt import *
from aml.time_mesuarment import timeit
from pprint import pprint as Print
from matplotlib import collections as matcoll
from numba import jit
from ParsingSystem.parse_and_build import TableGenerator

if __name__ == '__main__':
    to_new_var_name = torch.load(config.myocyte_map_from_old_param_name_to_new_name_dict_filename)
    params_vec,from_index_of_param_to_param_name = make_params_vec_from_params_dict(source_params_values=sysv1_de_config.params_values,
                                                  from_source_param_name_to_new=to_new_var_name)
    to_new_y_name = torch.load(config.myocyte_map_from_old_y_name_to_new_name_dict_filename)

    start_point = get_start_point_values(source_y_dict_with_start_point=sysv1_de_config.start_point,
                                         from_source_y_name_to_new=to_new_y_name)
    # y_vec_len = start_point.shape[0]
    # start_point = np.random.uniform(low=0.5, high=2.5, size=y_vec_len)


    tau = sysv1_de_config.tau
    t_0 = sysv1_de_config.t0
    t_end = sysv1_de_config.tend
    N = int((t_end-t_0)/tau)+1
    print('grid size {}'.format(N))
    time_grid = np.linspace(start=t_0, stop=t_end, num=N)
    solutions = odeint(func=F_vec, y0=start_point, t=time_grid, args=(params_vec,), full_output=False)

    # scipy.integrate.solve_ivp(fun, t_span, y0, method='RK45', t_eval=None, dense_output=False, events=None,
    #                           vectorized=False, args=None, **options)[source]
    fig = init_figure()
    fig = plot_solutions(fig, solutions, time_grid, to_new_y_name)
    FcarbTimeSeries = np.asarray([sysv1_de_config.F_carb(time_grid[i]) for i in range(len(time_grid))])
    FfatTimeSeries = np.asarray([sysv1_de_config.F_fat(time_grid[i]) for i in range(len(time_grid))])
    FprotTimeSeries = np.asarray([sysv1_de_config.F_prot(time_grid[i]) for i in range(len(time_grid))])
    add_line_to_fig(fig, time_grid,FprotTimeSeries,'F_{prot}')
    add_line_to_fig(fig, time_grid,FfatTimeSeries,'F_{fat}')
    add_line_to_fig(fig, time_grid,FcarbTimeSeries,'F_{carb}')

    fig.show()
    save_fig_to_html(fig,path=config.sysv1_path_to_solution,filename='solution.html')