import torch
import de_config
import config
from pprint import pprint
import numpy as np
from gen_des import F_vec
from scipy.integrate import odeint
import matplotlib.pyplot as plt

from gen_F import make_params_vec_from_params_dict, get_start_point_values, get_y_name_by_index_of_solution


def plot_solutions(solutions, names_dict):
    plt.rcParams["figure.figsize"] = [14, 7]
    plt.rcParams["figure.autolayout"] = True
    fig = plt.figure()
    axs = fig.add_subplot(111)
    number_of_solutions = solutions.shape[1]
    for i_of_sol in range(number_of_solutions):
        line_name = get_y_name_by_index_of_solution(index_of_solution= i_of_sol,
                                                    from_new_to_old_y_names_dict=names_dict)
        ith_solution = solutions[:, i_of_sol]
        axs.plot(time_grid, ith_solution, label = '$'+line_name+'$')
    axs.legend(bbox_to_anchor=(1.04, 1), loc="upper left",
               ncol=2,
               fancybox=True, shadow=True
               )
    axs.set_xlabel(r'$t$')
    axs.set_ylabel(r'$n$')
    axs.grid()
    plt.show()


if __name__ == '__main__':
    to_new_var_name = torch.load(config.map_from_old_param_name_to_new_name_dict_filename)
    params_vec = make_params_vec_from_params_dict(source_params_values=de_config.params_values,
                                                  from_source_param_name_to_new=to_new_var_name)
    to_new_y_name = torch.load(config.map_from_old_y_name_to_new_name_dict_filename)

    start_point = get_start_point_values(source_y_dict_with_start_point=de_config.start_point,
                                         from_source_y_name_to_new=to_new_y_name)
    time_grid = np.linspace(start=0.0, stop=5, num=10000)
    solutions = odeint(func=F_vec, y0=start_point, t=time_grid, args=(params_vec,))
    plot_solutions(solutions, to_new_y_name)

