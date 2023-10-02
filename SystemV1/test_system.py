from F_vec import *
from scipy.integrate import odeint
from myo_supportfs import *
from Plotting.myplt import *


index_by_name, name_by_index, start_point = get_start_point_names_mapping(start_point_dict)
# start_point = 0.1+ 10*np.random.rand(len(start_point))


BMR_process = {
    'INS': [start_point[index_by_name['INS']]],
    'times':[time_grid[0]],
    'T_a':[0.0],
     'INS_AUC_w':[0.0]
}
BMR_process['v_finder'] = ClosestValueFinder(BMR_process['times'],tau_grid, max(2, 2*tau_grid))
# solutions = odeint(func=F_vec, y0=start_point, t=time_grid ,args=(processes,BMR_process) ,full_output=False)
solutions = odeint(func=F_vec, y0=start_point, t=time_grid ,full_output=False)

intervals = get_intervals_of_processes(solutions, time_grid, index_by_name)

print(np.min(solutions),np.max(solutions))
# print(intervals['INS'])
# print(intervals['GLN_CAM'])
# print(intervals['GLN_INS_CAM'])

h_max = np.max(solutions)
h_min = np.min(solutions)
step_ = (h_max-h_min)/10

fig = init_figure(x_label=r'$t,min$',y_label=r'$\frac{mmol}{L}$')
fig = plot_solutions(fig, solutions, time_grid, name_by_index)

add_line_to_fig(fig, time_grid, np.array([J_fat_func(t) for t in time_grid]), r'Fat')
add_line_to_fig(fig, time_grid, np.array([J_prot_func(t) for t in time_grid]), r'Prot')
add_line_to_fig(fig, time_grid, np.array([J_carb_func(t) for t in time_grid]), r'Carb')

add_line_to_fig(fig, time_grid, np.array([J_flow_fat_func(t) for t in time_grid]), r'n_{TG}^{+}')
add_line_to_fig(fig, time_grid, np.array([J_flow_prot_func(t) for t in time_grid]), r'n_{AA}^{+}')
add_line_to_fig(fig, time_grid, np.array([J_flow_carb_func(t) for t in time_grid]), r'n_{Glu}^{+}')

fig = plot_intervals_to_plotly_fig(fig, intervals, 
                                   {    'INS': h_max,
                                        'GLN_CAM': h_max-step_,
                                        'GLN_INS_CAM': h_max-2*step_,
                                        'fasting':h_max-3*step_},
                                   {    'INS': "#FF0000",
                                        'GLN_CAM': "#7FFF00",
                                        'GLN_INS_CAM': "#87CEEB",
                                        'fasting':"#04e022"})


fig.show()