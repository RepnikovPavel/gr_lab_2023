import torch
import de_config
import config
from pprint import pprint
import numpy as np
from gen_des import F_vec
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objects as go
import seaborn as sns

from gen_F import make_params_vec_from_params_dict, get_start_point_values, get_y_name_by_index_of_solution


def plot_solutions(solutions, time_grid, names_dict):

    # https://towardsdatascience.com/8-visualizations-with-python-to-handle-multiple-time-series-data-19b5b2e66dd0

    # plotly (javascript)
    # 'plotly_dark','none'
    plotly.io.templates.default = 'plotly_dark'
    fig = go.Figure(
            layout=go.Layout(
            # title="Mt Bruno Elevation",
            xaxis_title='$t,s$',
            yaxis_title=r'$n, \frac{mol}{L}$',
            # xaxis=dict(rangeslider=dict(visible=True)), # add slider
            # width=1900, height=4000
            # margin=dict(
            #     l=0,
            #     r=0,
            #     b=0,
            #     t=0,
            #     pad=4
            # ),
        ))
    number_of_solutions = solutions.shape[1]
    for i_of_sol in range(number_of_solutions):

        line_name = get_y_name_by_index_of_solution(index_of_solution= i_of_sol,
                                                    from_new_to_old_y_names_dict=names_dict)
        ith_solution = solutions[:, i_of_sol]
        line_style = None
        if np.sum(np.where(ith_solution<0))>0:
            line_style ='dot'

        fig.add_trace(go.Scatter(x=time_grid,
                                 y=ith_solution,
                                 name='$'+line_name+'$',
                                 fill=None,
                                 line=dict(width=4, dash=line_style)
                                 )
                      )
    # fig.update_layout(xaxis_title='$t,s$',
    #                   yaxis_title='$n$')
    # save plotly figure to html
    # fig.write_html(config.plotly_plotting_html)
    # open figure in browser
    fig.show()

    # plotly multiple plots in one html
    # with open(config.plotly_plotting_html, 'w') as f:
    #     number_of_solutions = solutions.shape[1]
    #     for i_of_sol in range(number_of_solutions):
    #
    #
    #
    #         line_name = get_y_name_by_index_of_solution(index_of_solution= i_of_sol,
    #                                                     from_new_to_old_y_names_dict=names_dict)
    #
    #
    #         ith_solution = solutions[:, i_of_sol]
    #         line_style = None
    #         if np.sum(np.where(ith_solution<0))>0:
    #             line_style ='dot'
    #
    #         fig_i = go.Figure(
    #             layout=go.Layout(
    #                 # title="Mt Bruno Elevation",
    #                 xaxis_title='$t,s$',
    #                 yaxis_title=r'$n, \frac{mol}{L}$',
    #                 title='$'+line_name+'$',
    #                 # xaxis=dict(rangeslider=dict(visible=True)), # add slider
    #                 # width=1900, height=4000
    #                 # margin=dict(
    #                 #     l=0,
    #                 #     r=0,
    #                 #     b=0,
    #                 #     t=0,
    #                 #     pad=4
    #                 # ),
    #             ))
    #
    #         fig_i.add_trace(go.Scatter(x=time_grid,
    #                                  y=ith_solution,
    #                                  name='$'+line_name+'$',
    #                                  fill=None,
    #                                  line=dict(width=4, dash=line_style)
    #                                  )
    #                       )
    #
    #         f.write(fig_i.to_html(full_html=False, include_plotlyjs='cdn'))


    # matplotlib
    # plt.rcParams["figure.figsize"] = [14, 7]
    # plt.rcParams["figure.autolayout"] = True
    # fig = plt.figure()
    # axs = fig.add_subplot(111)
    # number_of_solutions = solutions.shape[1]
    # for i_of_sol in range(number_of_solutions):
    #
    #     line_name = get_y_name_by_index_of_solution(index_of_solution= i_of_sol,
    #                                                 from_new_to_old_y_names_dict=names_dict)
    #     ith_solution = solutions[:, i_of_sol]
    #     if np.sum(np.where(ith_solution<0))>0:
    #         axs.plot(time_grid, ith_solution, label = '$'+line_name+'$', linestyle='dashed')
    #     else:
    #         axs.plot(time_grid, ith_solution, label = '$'+line_name+'$')
    # axs.legend(bbox_to_anchor=(1.04, 1), loc="upper left",
    #            ncol=2,
    #            fancybox=True, shadow=True
    #            )
    # axs.set_xlabel(r'$t$')
    # axs.set_ylabel(r'$n$')
    # axs.grid()
    # plt.show()


if __name__ == '__main__':
    to_new_var_name = torch.load(config.map_from_old_param_name_to_new_name_dict_filename)
    params_vec = make_params_vec_from_params_dict(source_params_values=de_config.params_values,
                                                  from_source_param_name_to_new=to_new_var_name)
    to_new_y_name = torch.load(config.map_from_old_y_name_to_new_name_dict_filename)

    start_point = get_start_point_values(source_y_dict_with_start_point=de_config.start_point,
                                         from_source_y_name_to_new=to_new_y_name)
    time_grid = np.linspace(start=0.0, stop=5, num=10000)
    solutions = odeint(func=F_vec, y0=start_point, t=time_grid, args=(params_vec,))

    # scipy.integrate.solve_ivp(fun, t_span, y0, method='RK45', t_eval=None, dense_output=False, events=None,
    #                           vectorized=False, args=None, **options)[source]


    plot_solutions(solutions,time_grid, to_new_y_name)

