from F_vec import *
from scipy.integrate import odeint
from myo_supportfs import *
from Plotting.myplt import *

index_by_name, name_by_index, start_point = get_start_point_names_mapping(start_point_dict)
N = int((t_end-t_0)/tau_grid)+1
time_grid = np.linspace(start=t_0, stop=t_end, num=N)
solutions = odeint(func=F_vec, y0=start_point, t=time_grid , full_output=False)

fig = init_figure()
fig = plot_solutions(fig, solutions, time_grid, name_by_index)
fig.show()