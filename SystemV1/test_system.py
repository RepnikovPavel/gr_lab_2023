from F_vec import *
from scipy.integrate import odeint
from myo_supportfs import *
from Plotting.myplt import *


index_by_name, name_by_index, start_point = get_start_point_names_mapping(start_point_dict)
# start_point = 0.1+ 10*np.random.rand(len(start_point))

processes = {
    'time_point':[],
    'INS': [],
    'GLN_CAM': [],
    'GLN_INS_CAM': []
}
solutions = odeint(func=F_vec, y0=start_point, t=time_grid ,args=(processes,) ,full_output=False)
intervals = get_intervals_of_processes(processes)
print(np.min(solutions),np.max(solutions))
# print(intervals['INS'])
# print(intervals['GLU_CAM'])
# print(intervals['GLU_INS_CAM'])

h_max = np.max(solutions)
h_min = np.min(solutions)
step_ = (h_max-h_min)/10

fig = init_figure(x_label=r'$t,min$',y_label=r'$\frac{mmol}{L}$')
fig = plot_solutions(fig, solutions, time_grid, name_by_index)
fig = plot_intervals_to_plotly_fig(fig, intervals, 
                                   {    'INS': h_max,
                                        'GLN_CAM': h_max-step_,
                                        'GLN_INS_CAM': h_max-2*step_},
                                   {    'INS': "#FF0000",
                                        'GLN_CAM': "#7FFF00",
                                        'GLN_INS_CAM': "#87CEEB"})
fig.show()