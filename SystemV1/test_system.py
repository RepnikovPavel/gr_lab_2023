from F_vec import *
from scipy.integrate import odeint
from myo_supportfs import *
from Plotting.myplt import *
from pprint import pprint
from scipy.integrate import ode,solve_ivp


index_by_name, name_by_index, start_point = get_start_point_names_mapping(start_point_dict)
# start_point = 0.1+ 10*np.random.rand(len(start_point))


INS_on_grid = np.zeros(shape=(len(time_grid), ),dtype=np.float32)
INS_AUC_w_on_grid = np.zeros(shape=(len(time_grid), ),dtype=np.float32)
INS_on_grid[0] = start_point[index_by_name['INS']]
T_a_on_grid = np.zeros(shape=(len(time_grid), ),dtype=np.float32)
INS_AUC_w_on_grid[0] = 0.0
T_a_on_grid[0]= 0.0
last_seen_time = np.zeros(shape=(1,),dtype=np.float32)
last_seen_time[0] = t_0
last_time_pos = np.zeros(shape=(1,),dtype=np.intc)
last_time_pos[0] = 0

def F_wrapped(t, y):
    return F_vec(t,y,INS_on_grid,INS_AUC_w_on_grid,T_a_on_grid,last_seen_time,last_time_pos)

solver = ode(f=F_wrapped,jac=None)

# solver.set_f_params(INS_on_grid,INS_AUC_w_on_grid,T_a_on_grid,last_seen_time,last_time_pos)
solver.set_initial_value(y=start_point,t=t_0)
# solver.set_integrator('vode',method='bdf') 
# method = 'lsoda'
# method = 'dopri5'
method = 'vode'
solver.set_integrator(method) 
solutions = np.zeros(shape=(len(time_grid),len(start_point)),dtype=np.float32)
solutions[0,:] = solver.y
i_=  1
while solver.successful() and solver.t < t_end-tau_grid:
    solutions[i_,:] = solver.integrate(solver.t+tau_grid)
    i_ += 1 
print('last solver time step {} target last step {}'.format(i_, len(time_grid)))
time_sol = time_grid

# output = odeint(func=F_vec, y0=start_point, t=time_grid, args=(INS_on_grid,INS_AUC_w_on_grid,T_a_on_grid, 
#                                                                   last_seen_time,last_time_pos),full_output=1)
                                                                 #  hmin=0.001,hmax=1.0)

# solutions = output[0]
# solver_o = output[1]
# pprint(solver_o)
# def F_wrapped(t,y):
#     return F_vec(t,y,INS_on_grid,INS_AUC_w_on_grid,T_a_on_grid, last_seen_time,last_time_pos) 
# F_wrapped = lambda t,y: 
# sol = solve_ivp(fun=F_vec,t_span=(t_0,t_end),y0=start_point, args=(INS_on_grid,INS_AUC_w_on_grid,T_a_on_grid, last_seen_time,last_time_pos))
# sol = solve_ivp(fun=F_vec,t_span=(t_0,t_end),y0=start_point,args=(INS_on_grid,INS_AUC_w_on_grid,T_a_on_grid, 
#                                                                   last_seen_time,last_time_pos))
# time_sol =  sol.t
# solutions = sol.y.T
pprint(solutions.shape)

intervals = get_intervals_of_processes(solutions, time_grid, index_by_name)

print(np.min(solutions),np.max(solutions))
# print(intervals['INS'])
# print(intervals['GLN_CAM'])
# print(intervals['GLN_INS_CAM'])

h_max = 120
h_min = 100
step_ = (h_max-h_min)/10

fig = init_figure(x_label=r'$t,min$',y_label=r'$\frac{mmol}{L}$')
fig = plot_solutions(fig, solutions, time_grid, name_by_index)

add_line_to_fig(fig, time_grid, np.array([J_fat_func(t) for t in time_grid]), r'Fat')
add_line_to_fig(fig, time_grid, np.array([J_prot_func(t) for t in time_grid]), r'Prot')
add_line_to_fig(fig, time_grid, np.array([J_carb_func(t) for t in time_grid]), r'Carb')

add_line_to_fig(fig, time_grid, np.array([J_flow_fat_func(t) for t in time_grid]), r'J_{TG}^{+}')
add_line_to_fig(fig, time_grid, np.array([J_flow_prot_func(t) for t in time_grid]), r'J_{AA}^{+}')
add_line_to_fig(fig, time_grid, np.array([J_flow_carb_func(t) for t in time_grid]), r'J_{Glu}^{+}')

add_line_to_fig(fig, time_grid, T_a_on_grid, r'T_{a}')
add_line_to_fig(fig, time_grid, INS_AUC_w_on_grid, r'AUC_{w}(INS)')
# add_line_to_fig(fig, time_grid, solver_o[''], r'')



fig = plot_intervals_to_plotly_fig(fig, intervals, 
                                   {    'INS': h_min+step_,
                                        'GLN_CAM': h_min+step_*2,
                                        'GLN_INS_CAM': h_min+step_*3,
                                        'fasting':h_min+step_*4},
                                   {    'INS': "#FF0000",
                                        'GLN_CAM': "#7FFF00",
                                        'GLN_INS_CAM': "#87CEEB",
                                        'fasting':"#04e022"})


fig.show()