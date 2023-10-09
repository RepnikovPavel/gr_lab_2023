from F_vec import *
from scipy.integrate import odeint
from myo_supportfs import *
from Plotting.myplt import *
from pprint import pprint
from scipy.integrate import ode,solve_ivp
from euler import euler_solver


index_by_name, name_by_index, start_point = get_start_point_names_mapping(start_point_dict)
# start_point = 0.1+ 10*np.random.rand(len(start_point))

# J_flow_carb_vs = np.zeros(shape=(len(J_flow_carb_func.values)),dtype=np.float32)
J_flow_prot_vs = np.zeros(shape=(len(J_flow_prot_func.values)),dtype=np.float32)
J_flow_fat_vs  = np.zeros(shape=(len(J_flow_fat_func.values)),dtype=np.float32) 

J_flow_carb_vs = J_flow_carb_func.values
# J_flow_prot_vs = J_flow_prot_func.values
# J_flow_fat_vs  = J_flow_fat_func.values 

# AUC auxiliary arrays
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

# BMR auxiliary arrays
J_KB_plus = np.zeros(shape=(len(time_grid),),dtype=np.float32)
J_AA_minus = np.zeros(shape=(len(time_grid),),dtype=np.float32)
J_Glu_minus = np.zeros(shape=(len(time_grid),),dtype=np.float32)
J_FFA_minus = np.zeros(shape=(len(time_grid),),dtype=np.float32)
J_KB_minus = np.zeros(shape=(len(time_grid),),dtype=np.float32)



def F_wrapped(t, y):
    return F_vec(t,y,INS_on_grid,INS_AUC_w_on_grid,T_a_on_grid,last_seen_time,last_time_pos,
                 J_flow_carb_vs,
                    J_flow_prot_vs,
                    J_flow_fat_vs,
                    J_KB_plus,
                    J_AA_minus,
                    J_Glu_minus,
                    J_FFA_minus,
                    J_KB_minus)

solver = ode(f=F_wrapped,jac=None)
solver.set_initial_value(y=start_point,t=t_0)
# solver_type = 'lsoda'
# solver_type = 'dopri5'
solver_type = 'vode'
solver.set_integrator(solver_type) 
solutions = np.zeros(shape=(len(time_grid),len(start_point)),dtype=np.float32)
solutions[0,:] = solver.y
i_=  1
while solver.successful() and solver.t < t_end:
# while solver.t < t_end-tau_grid:
    solutions[i_,:] = solver.integrate(solver.t+tau_grid)
    i_ += 1 
print('last solver time step {} target last step {}'.format(i_, len(time_grid)-1))
time_sol = time_grid

# output = odeint(tfirst=True,func=F_wrapped, y0=start_point, t=time_grid,full_output=1)
# solutions = output[0]
# time_sol = time_grid
# solver_o = output[1]

# solutions = euler_solver(func=F_wrapped, y0=start_point, t=time_grid,name_by_index=name_by_index,debug=False)
# time_sol = time_grid

# sol = solve_ivp(fun=F_wrapped,t_span=(t_0,t_end),y0=start_point,t_eval=time_grid,method='Radau')
# solutions = sol.y.T
# time_sol = sol.t
# print(sol.message)


print(solutions.shape)
print(time_sol.shape)    

# for i in range(len(solutions[0])):
#      where_nan_i = np.argwhere(np.isnan(solutions[:,i]))
#      if len(where_nan_i) != 0 :
#           print(name_by_index[i])
# intervals = get_intervals_of_processes(solutions, time_sol, index_by_name)
intervals = {}


h_max = np.max(solutions)
h_min = np.min(solutions)
print(h_min,h_max)

step_ = (h_max-h_min)/10

fig = init_figure(x_label=r'$t,min$',y_label=r'$\frac{mmol}{L}$')
fig = plot_solutions(fig, solutions, time_sol, name_by_index)

add_line_to_fig(fig, time_grid, np.array([J_fat_func(t) for t in time_grid]), r'Fat')
add_line_to_fig(fig, time_grid, np.array([J_prot_func(t) for t in time_grid]), r'Prot')
add_line_to_fig(fig, time_grid, np.array([J_carb_func(t) for t in time_grid]), r'Carb')

add_line_to_fig(fig, time_grid, np.array([J_flow_fat_func(t) for t in time_grid]), r'J_{TG}^{+}')
add_line_to_fig(fig, time_grid, np.array([J_flow_prot_func(t) for t in time_grid]), r'J_{AA}^{+}')
add_line_to_fig(fig, time_grid, np.array([J_flow_carb_func(t) for t in time_grid]), r'J_{Glu}^{+}')

add_line_to_fig(fig, time_grid, T_a_on_grid, r'T_{a}')
add_line_to_fig(fig, time_grid, INS_AUC_w_on_grid, r'AUC_{w}(INS)')

add_line_to_fig(fig, time_grid, J_KB_minus, r'J^{-}_{KB}')
add_line_to_fig(fig, time_grid, J_KB_plus, r'J^{+}_{KB}')
add_line_to_fig(fig, time_grid, J_AA_minus, r'J^{-}_{AA}')
add_line_to_fig(fig, time_grid, J_FFA_minus, r'J^{-}_{FFA}')
add_line_to_fig(fig, time_grid, J_Glu_minus, r'J^{-}_{Glu}')

add_line_to_fig(fig,time_sol, EnergyOnGrid(AA=solutions[:,index_by_name['AA_ef']],
                                           FFA=solutions[:,index_by_name['FFA_ef']],
                                           KB=solutions[:,index_by_name['KB_ef']],
                                           Glu=solutions[:,index_by_name['Glu_ef']],
                                           beta_AA=beta_AA,beta_FFA=beta_FFA,beta_KB=beta_KB,beta_Glu=beta_Glu),
                r'E_{system}[kkal]')


fig = plot_intervals_to_plotly_fig(fig, intervals, 
                                   {    'INS': h_max-step_,
                                        'GLN_CAM': h_max-step_*2,
                                        'GLN_INS_CAM': h_max-step_*3,
                                        'fasting':h_max-step_*4},
                                   {    'INS': "#FF0000",
                                        'GLN_CAM': "#7FFF00",
                                        'GLN_INS_CAM': "#87CEEB",
                                        'fasting':"#04e022"})

fig.show()