from F_vec import *
from scipy.integrate import odeint
from myo_supportfs import *
from Plotting.myplt import *

index_by_name, name_by_index, start_point = get_start_point_names_mapping(start_point_dict)
start_point = 0.1+ 10*np.random.rand(len(start_point))
solutions = odeint(func=F_vec, y0=start_point, t=time_grid , full_output=False)

fig = init_figure(x_label=r'$t,min$',y_label=r'$\frac{mmol}{L}$')
fig = plot_solutions(fig, solutions, time_grid, name_by_index)
fig.show()