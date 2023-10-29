import numpy as np
import numba 
from input import *
import torch
from local_contributor_config import problem_folder

W =  50.0 # [min] window to check AUC(INS, t-W,t)
INS_check_coeff = 400 # [mmol*s]

# take grid on from FPC.ipynb file 
tau_grid = 0.01 # [min]
t_0 = 400.0 # [min]
t_end = 3000.0 # [min]
# t_end = 500.0 # [min]
t_0_input= 0.0
tau_grid_input = 0.1

N = int((t_end-t_0)/tau_grid)+1
time_grid = np.linspace(start=t_0, stop=t_end, num=N)

# make input data
J_flow_prot_func = torch.load(
    os.path.join(problem_folder, 'ddt_AA_ef'))

J_flow_carb_func = torch.load(
    os.path.join(problem_folder, 'ddt_Glu_ef'))
J_flow_fat_func = torch.load(
    os.path.join(problem_folder, 'ddt_TG_pl'))
J_prot_func = torch.load(
    os.path.join(problem_folder, 'J_prot'))

J_fat_func = torch.load(
    os.path.join(problem_folder, 'J_fat'))
J_carb_func = torch.load(
    os.path.join(problem_folder, 'J_carb'))

beta_KB = 517.0/1000.0 # [kcal/mmol]
beta_Glu = 699.0/1000.0 # [kcal/mmol]
beta_AA = 369.5/1000.0 # [kcal/mmol]
beta_FFA = 2415.6/1000.0 # [kcal/mmol]

inv_beta_KB = 1.0/(517.0/1000.0) # 1/[kcal/mmol]
inv_beta_Glu = 1.0/(699.0/1000.0) # 1/[kcal/mmol]
inv_beta_AA = 1.0/(369.5/1000.0) # 1/[kcal/mmol]
inv_beta_FFA = 1.0/(2415.6/1000.0) # 1/[kcal/mmol]


MASS_OF_HUMAN = 70.0
E_day = 1500.0 # [kcal/day]
e_sigma = E_day/(24.0*60.0) #[kcal/min]

power_of_coeff = -1


# IF (есть лишние AA) THEN (rest_cont идет на расход AA)
# IF (нет лишних AA AND есть Glu AND есть INS) THEN (rest_cont идет на расход Glu)
# IF (нет инсулина INS AND нет инсулина <= 180 [мин]) THEN (rest_cont идет на расход FFA)
# IF (нет инсулина INS AND нет инсулина > 180 [мин]) THEN (rest_cont идет на расход FFA AND rest_cont идет на расход KB )
# IF (нет инсулина INS AND нет инсулина > 180 [мин]) THEN (рост кетоновых тел v=BMR*(0.5/100.0) [kcal/hour])
# IF (7*60[min] голодания) THEN (расход KB 0.07 + 0.01*7 [kcal/min] v=BMR*(7.0/100.0))
# IF (70*60[min] голодания) THEN (расход KB 0.07 + 0.01*70 [kcal/min]v=BMR*(38.5/100.0))



lambda_ = 1.0
sigma = 0.07

alpha_base       = 2.0
beta_base=         0.02
gamma_base =        1.0

CL_GLN_base=1.0/10.0
CL_CAM_base=1.0/10.0
CL_INS_base=1.0/10.0

# коэффициенты, отвечающие за перекачку энергии из Glu,FFA,KB,AA из крови
m_1_base=            1.0
m_3_base=            1.0
m_4_base=            1.0
m_5_base=            1.0


# номера коэффициентов
a_1_base=            10.0**(power_of_coeff)
a_2_base=            10.0**(power_of_coeff)
a_3_base=            10.0**(power_of_coeff)
a_4_base=            10.0**(power_of_coeff)
a_5_base=            10.0**(power_of_coeff)
a_6_base=            10.0**(power_of_coeff)
a_7_base=            10.0**(power_of_coeff)
a_8_base=            10.0**(power_of_coeff)
a_9_base=            10.0**(power_of_coeff)
a_10_base=            10.0**(power_of_coeff)
a_11_base=            10.0**(power_of_coeff)
a_12_base=            10.0**(power_of_coeff)
a_13_base=            10.0**(power_of_coeff)
a_14_base=            10.0**(power_of_coeff)
a_15_base=            10.0**(power_of_coeff)
a_16_base=            10.0**(power_of_coeff)
a_17_base=            10.0**(power_of_coeff)
a_18_base=            10.0**(power_of_coeff)
a_19_base=            10.0**(power_of_coeff)
m_2_base=            10.0**(power_of_coeff)
m_6_base=            10.0**(power_of_coeff)
m_7_base=            10.0**(power_of_coeff)
m_8_base=            10.0**(power_of_coeff)
m_9_base=            10.0**(power_of_coeff)
m_10_base=           10.0**(power_of_coeff)
m_11_base=           10.0**(power_of_coeff)
m_12_base=           10.0**(power_of_coeff)  # *[Carnitin]
m_13_base=           10.0**(power_of_coeff)
m_14_base=           10.0**(power_of_coeff)
m_15_base=           10.0**(power_of_coeff)
m_16_base=           10.0**(power_of_coeff) # *[Creatin]
m_17_base=           10.0**(power_of_coeff)
m_18_base=           10.0**(power_of_coeff)
m_19_base=           10.0**(power_of_coeff)
m_20_base=           10.0**(power_of_coeff)
m_21_base=           10.0**(power_of_coeff)
h_1_base=            10.0**(power_of_coeff)
h_2_base=            10.0**(power_of_coeff)
h_3_base=            10.0**(power_of_coeff)
h_4_base=            10.0**(power_of_coeff)
h_5_base=            10.0**(power_of_coeff)
h_6_base=            10.0**(power_of_coeff)
h_7_base=            10.0**(power_of_coeff)
h_8_base=            10.0**(power_of_coeff)
h_9_base=            10.0**(power_of_coeff)
h_10_base=            10.0**(power_of_coeff)
h_11_base=            10.0**(power_of_coeff)
h_12_base=            10.0**(power_of_coeff)
h_13_base=            10.0**(power_of_coeff)
h_14_base=            10.0**(power_of_coeff)
h_15_base=            10.0**(power_of_coeff)
h_16_base=            10.0**(power_of_coeff)
h_17_base=            10.0**(power_of_coeff)
h_18_base=            10.0**(power_of_coeff)
h_19_base=            10.0**(power_of_coeff)
h_20_base=            10.0**(power_of_coeff)
h_21_base=            10.0**(power_of_coeff)
h_22_base=            10.0**(power_of_coeff)
h_23_base=            10.0**(power_of_coeff)
h_24_base=            10.0**(power_of_coeff)
h_25_base=            10.0**(power_of_coeff)
h_26_base=            10.0**(power_of_coeff)
h_27_base=            10.0**(power_of_coeff)
h_28_base=            10.0**(power_of_coeff)
h_29_base=            10.0**(power_of_coeff)

j_0_base = 0.0
j_1_base = 0.0
j_2_base = 0.0
j_3_base = 0.0
j_4_base = 0.0

Glu_ef_start= E_day/beta_Glu/4
AA_ef_start = E_day/beta_AA/4
FFA_ef_start = E_day/beta_FFA/4
KB_ef_start = E_day/beta_KB/4

start_point_dict = {
    'Glu_ef':Glu_ef_start,
    'AA_ef':AA_ef_start,
    'Glycerol_ef':10.0,
    'FFA_ef':FFA_ef_start,
    'Lac_m':10.0,
    'KB_ef':KB_ef_start,
    'Cholesterol_pl':10.0,
    'TG_pl':10.0,
    'G6_a':10.0,
    'G3_a':10.0,
    'Pyr_a':10.0,
    'Ac_CoA_a':10.0,
    'FA_CoA_a':10.0,
    'Cit_a':10.0,
    'OAA_a':10.0,
    'AA_a':10.0,
    'NADPH_a':10.0,
    'TG_a':10.0,
    'GG_m':10.0,
    'G6_m':10.0,
    'G3_m':10.0,
    'Pyr_m':10.0,
    'Ac_CoA_m':10.0,
    'FA_CoA_m':10.0,
    'Cit_m':10.0,
    'OAA_m':10.0,
    'H_cyt_m':10.0,
    'H_mit_m':10.0,
    'AA_m':10.0,
    'Muscle_m':10.0,
    'CO2_m':10.0,
    'H2O_m':10.0,
    'ATP_cyt_m':10.0,
    'ATP_mit_m':10.0,
    'GG_h':10.0,
    'G6_h':10.0,
    'G3_h':10.0,
    'Pyr_h':10.0,
    'Ac_CoA_h':10.0,
    'FA_CoA_h':10.0,
    'MVA_h':10.0,
    'Cit_h':10.0,
    'OAA_h':10.0,
    'NADPH_h':10.0,
    'AA_h':10.0,
    'TG_h':10.0,
    'INS':10.0,
    'CAM':10.0,
    'GLN':10.0,
    'Urea_ef':10.0,
}




# HeartRate_func = HeartRate_gen(tau_grid,time_grid,60,180)
  

# HR_vs = HeartRate_func.values

# def F_vec(y_vec: np.array,t: float,processes, BMR_process):

@jit(nopython = True)
def F_vec(t: float, y_vec: np.array,
          INS_on_grid:np.array, INS_AUC_w_on_grid:np.array,T_a_on_grid:np.array,
          last_seen_time:np.array,last_time_pos:np.array,
            J_flow_carb_vs:np.array,
            J_flow_prot_vs:np.array,
            J_flow_fat_vs:np.array,                    
            J_KB_plus_arr:np.array,
            J_AA_minus_arr:np.array,
            J_Glu_minus_arr:np.array,
            J_FFA_minus_arr:np.array,
            J_KB_minus_arr:np.array):
    buffer = np.zeros(shape=(50, ),dtype=np.float32)
    # свободные функции 
    # J_carb_flow = J_flow_carb_func(t)
    # J_prot_flow = J_flow_prot_func(t)
    # J_fat_flow  = J_flow_fat_func(t)
    # HeartRate = HeartRate_func(t)
    # print(t)
    time_index_i = np.intc((t-t_0_input)/tau_grid_input)
    J_carb_flow = J_flow_carb_vs[time_index_i]
    J_prot_flow = J_flow_prot_vs[time_index_i]
    J_fat_flow  = J_flow_fat_vs[time_index_i]
    t_pos = np.maximum(np.intc(0), np.intc((t-t_0)/tau_grid))
    # HeartRate = HR_vs[t_pos]
    HeartRate = 80.0

    # Y_{t} values
    # значения в момент времени t
    y0 = y_vec[0]                  
    y1 = y_vec[1]                   
    y2 = y_vec[2]             
    y3 = y_vec[3]                 
    y4 = y_vec[4]                   
    y5 = y_vec[5]                  
    y6  = y_vec[6]           
    y7 = y_vec[7]                   
    y8 = y_vec[8]                    
    y9 = y_vec[9]            
    y10 = y_vec[10]           
    y11 = y_vec[11]        
    y12 = y_vec[12]        
    y13 = y_vec[13]           
    y14 = y_vec[14]           
    y15 = y_vec[15]            
    y16 = y_vec[16]         
    y17 = y_vec[17]                     
    y18 = y_vec[18]                     
    y19 = y_vec[19]            
    y20 = y_vec[20]            
    y21 = y_vec[21]           
    y22 = y_vec[22]        
    y23 = y_vec[23]        
    y24 = y_vec[24]           
    y25 = y_vec[25]           
    y26 = y_vec[26]         
    y27 = y_vec[27]         
    y28 = y_vec[28]            
    y29 = y_vec[29]                 
    y30 = y_vec[30]           
    y31 = y_vec[31]           
    y32 = y_vec[32]        
    y33 = y_vec[33]        
    y34 = y_vec[34]                    
    y35 = y_vec[35]            
    y36 = y_vec[36]            
    y37 = y_vec[37]           
    y38 = y_vec[38]        
    y39 = y_vec[39]        
    y40 = y_vec[40]           
    y41 = y_vec[41]           
    y42 = y_vec[42]           
    y43 = y_vec[43]         
    y44 = y_vec[44]            
    y45 = y_vec[45]
    y46 = y_vec[46]
    y47 = y_vec[47]
    y48 = y_vec[48]            
    y49 = y_vec[49]                 

    insulin_activation_coefficient =  15.0
    is_insulin_process = Heviside(y46-insulin_activation_coefficient)
    # is_insulin_process = Sigmoid(y46-insulin_activation_coefficient)
    a_2 = is_insulin_process * a_2_base
    a_4 = is_insulin_process * a_4_base
    a_7 = is_insulin_process * a_7_base
    m_1 = is_insulin_process * m_1_base
    m_7 = is_insulin_process * m_7_base
    h_3 = is_insulin_process * h_3_base
    h_10 = is_insulin_process * h_10_base
    h_19 = is_insulin_process * h_19_base
    h_20 = is_insulin_process * h_20_base

    h_12 = h_12_base
    h_24 = h_24_base
    h_17 = h_17_base
    h_16 = h_16_base
    h_26 = h_26_base
    h_7 = h_7_base
    j_0 = j_0_base
    a_5 = a_5_base
    a_13 = a_13_base
    a_14 = a_14_base
    a_10 = a_10_base
    a_12 = a_12_base
    m_9 = m_9_base
    m_11 = m_11_base

    # glucagon_adrenilin_activation_coefficient = y48+y47
    # is_glucagon_adrenalin_process = Heviside(glucagon_adrenilin_activation_coefficient-160.0)
    # is_glucagon_adrenalin_process = Sigmoid(glucagon_adrenilin_activation_coefficient-160.0)
    # h_23 = is_glucagon_adrenalin_process * h_23_base
    # h_18 = is_glucagon_adrenalin_process * h_18_base 
    # h_13 = is_glucagon_adrenalin_process * h_13_base
    # h_2 = is_glucagon_adrenalin_process *  h_2_base
    # a_9 = is_glucagon_adrenalin_process *  a_9_base


    # is_glucagon_adrenalin_insulin_process = Heviside(glucagon_adrenalin_insulin_activation_coefficient-1.0)
    is_glucagon_adrenalin_insulin_process = 1.0
    # is_glucagon_adrenalin_insulin_process = Sigmoid(glucagon_adrenalin_insulin_activation_coefficient-1.0)
    h_11 = is_glucagon_adrenalin_insulin_process * h_11_base 
    h_25 = is_glucagon_adrenalin_insulin_process * h_25_base
    h_6 = is_glucagon_adrenalin_insulin_process * h_6_base
    a_3 = is_glucagon_adrenalin_insulin_process * a_3_base
    a_11 = is_glucagon_adrenalin_insulin_process * a_11_base
    m_8 = is_glucagon_adrenalin_insulin_process * m_8_base
    

    glucagon_adrenalin_activation_coefficient = (y48+y47)
    is_glucagon_adrenalin_process = Heviside(y48+y47 - 156.0)
    h_23 = is_glucagon_adrenalin_process * h_23_base
    h_18 = is_glucagon_adrenalin_process * h_18_base 
    h_13 = is_glucagon_adrenalin_process * h_13_base
    h_2 = is_glucagon_adrenalin_process *  h_2_base
    a_9 = is_glucagon_adrenalin_process *  a_9_base

    a_3 = is_glucagon_adrenalin_process * a_3_base
    m_8 = is_glucagon_adrenalin_process * m_8_base 
    h_12 = is_glucagon_adrenalin_process * h_2_base
    h_11 = is_glucagon_adrenalin_process * h_11_base
    h_13 = is_glucagon_adrenalin_process * h_13_base

    # AUC 

    AUC_at_t = -1.0
    T_a_t = -1.0

    # print(last_time_pos[0],t_pos)
    if t_pos - last_time_pos[0] > 0:
        diff_ = t_pos - last_time_pos[0]
        T_a_current = T_a_on_grid[last_time_pos[0]]
        last_seen_time[0] = t 
        t_minus_w_pos = np.maximum(np.intc(0), np.intc((t-W-t_0)/tau_grid))

        for j in range(1,diff_+1):
            INS_on_grid[last_time_pos[0]+j] = y46

        AUC_at_t = AUC_at_linear_grid(tau_grid, INS_on_grid, t_minus_w_pos, t_pos)

        for j in range(1,diff_+1):
            INS_AUC_w_on_grid[last_time_pos[0]+j] = AUC_at_t

        if AUC_at_t < INS_check_coeff and (t-t_0) >= W:
            for j in range(1,diff_+1):
                T_a_on_grid[last_time_pos[0]+j] = T_a_current + tau_grid*j
        else:
            for j in range(1,diff_+1):
                T_a_on_grid[last_time_pos[0]+j] = 0.0
        T_a_t = T_a_current + tau_grid*diff_
        last_time_pos[0] += diff_
    else:
        # already seen time point. get AUC and T_{a}
        AUC_at_t = INS_AUC_w_on_grid[t_pos]
        T_a_t = T_a_on_grid[t_pos]

    # BMR
    e_AA_min = 0.1*e_sigma
    e_Glu_min = 0.2*e_sigma
    e_FFA_min = 0.035*e_sigma
    e_KB_min = 0.0

    e_AA_minus = 0.0
    e_Glu_minus = 0.0
    e_FFA_minus = 0.0
    e_KB_minus = 0.0

    if y1 >= 20.0:
        e_AA_minus = e_sigma - (e_Glu_min+e_KB_min+e_FFA_min)
        e_Glu_minus = e_Glu_min
        e_FFA_minus = e_FFA_min
        e_KB_minus = e_KB_min
    elif (y1 < 20.0) and (y0>=20.0) and (T_a_t==0.0):
        e_AA_minus = e_AA_min
        e_Glu_minus = e_sigma - (e_AA_min+e_KB_min+e_FFA_min)
        e_FFA_minus = e_FFA_min
        e_KB_minus = e_KB_min
    elif (T_a_t>0.0) and (T_a_t< 3*60.0):
        e_AA_minus = e_AA_min
        e_Glu_minus = e_Glu_min
        e_FFA_minus = e_sigma - (e_AA_min+e_Glu_min+e_KB_min)
        e_KB_minus = e_KB_min 
    elif T_a_t >= 3*60.0: 
        e_AA_minus = e_AA_min
        e_Glu_minus = e_Glu_min
        rest_coeff = e_sigma - (e_AA_min+e_Glu_min+e_FFA_min+e_KB_min)
        coeff1 = rest_coeff*0.5
        coeff2 = rest_coeff*0.5
        e_FFA_minus = e_FFA_min+coeff1
        e_KB_minus = e_KB_min+coeff2

    # e_KB_plus = 0.005*e_sigma
    # J_KB_plus = e_KB_plus*inv_beta_KB*Heviside(T_a_t-180.0)
    # J_AA_minus  = e_AA_minus*inv_beta_AA*Heviside(y1-100.0*e_AA_minus*inv_beta_AA)
    # J_Glu_minus  = e_Glu_minus*inv_beta_Glu*Heviside(y0-100.0*e_Glu_minus*inv_beta_Glu)
    # J_FFA_minus  = e_FFA_minus*inv_beta_AA*Heviside(y3-100.0*e_FFA_minus*inv_beta_FFA)
    # J_KB_minus  =  e_KB_minus*inv_beta_KB*Heviside(y5-100.0*e_KB_minus*inv_beta_KB)

    # J_KB_plus_arr[t_pos] = J_KB_plus
    # J_AA_minus_arr[t_pos] = J_AA_minus
    # J_Glu_minus_arr[t_pos] = J_Glu_minus
    # J_FFA_minus_arr[t_pos] = J_FFA_minus
    # J_KB_minus_arr[t_pos] = J_KB_minus

    
    J_AA_minus  = 0.0
    J_Glu_minus  = 0.0
    J_FFA_minus  = 0.0
    J_KB_minus  =  0.0
    J_KB_plus = 0.0

    m_2 = m_2_base 
    m_3 = m_3_base 
    m_4 = m_4_base 
    m_5 = m_5_base 
    m_6 = m_6_base 
    m_10 = m_10_base


    m_12 = m_12_base
    m_13 = m_13_base
    m_14 = m_14_base
    m_15 = m_15_base
    m_16 = m_16_base
    m_17 = m_17_base
    m_18 = m_18_base
    m_19 = m_19_base
    m_20 = m_20_base
    m_21 = m_21_base

    a_1 = a_1_base
    a_6 = a_6_base
    a_8 = a_8_base

    a_15 = a_15_base
    a_16 = a_16_base
    a_17 = a_17_base
    a_18 = a_18_base
    a_19 = a_19_base

    h_1 = h_1_base
    h_4 = h_4_base
    h_5 = h_5_base
    h_8 = h_8_base
    h_9 = h_9_base
    h_14 = h_14_base
    h_15 = h_15_base
    h_21 = h_21_base
    h_22 = h_22_base
    h_27 = h_27_base
    h_28 = h_28_base
    h_29 = h_29_base

    j_1 = j_1_base
    j_2 = j_2_base
    j_3 = j_3_base
    j_4 = j_4_base

    # 2. Myocyte
    M_1 = m_1 * y0
    M_2 = m_2 * y21 * y26
    M_3 = m_3 * y5
    M_4 = m_4 * y3
    M_5 = m_5 * y1
    M_6 = m_6 * y28
    M_7 = m_7 * y19
    M_8 = m_8 * y18
    M_9 = m_9 * y19
    M_10 = m_10 * y20
    M_11 = m_11 * y21
    M_12 = m_12 * y23
    M_13 = m_13 * y22 * y25
    M_14 = m_14 * y24
    M_15 = m_15 * y26
    M_16 = m_16 * y27 # * [O2]
    M_17 = m_17 * y28
    M_18 = m_18 * y28
    M_19 = m_19 * y28
    M_20 = m_20 * y28
    M_21 = m_21 * y29
    #3. Adipocyte
    A_1=a_1 * y1
    A_2=a_2 * y3
    A_3=a_3 * y17
    A_4=a_4 * y0
    A_5=a_5 * y8
    A_6=a_6 * y8
    A_7=a_7 * y9 * y12
    A_8=a_8 * y9
    A_9=a_9 * y14
    A_10=a_10 * y10
    A_11=a_11 * y10
    A_12=a_12 * y14
    A_13=a_13 * y11 * y16
    A_14=a_14 * y13
    A_15=a_15 * y13
    A_16=a_16 * y14 * y11
    A_17=a_17 * y15
    A_18=a_18 * y15
    A_19=a_19 * y15
    #4. Hepatocyte
    H_1=h_1 * y1
    H_2=h_2 * y35
    H_3=h_3 * y0
    H_4=h_4 * y2
    H_5=h_5 * y4
    H_6=h_6 * y38
    H_7=h_7 * y40
    H_8=h_8 * y3
    H_9=h_9 * y45
    H_10=h_10 * y35
    H_11=h_11 * y34
    H_12=h_12 * y35
    H_13=h_13 * y36
    H_14=h_14 * y35
    H_15=h_15 * y36
    H_16=h_16 * y37
    H_17=h_17 * y38 * y43
    H_18=h_18 * y39
    H_19=h_19 * y38 * y43
    H_20=h_20 * y36 * y39
    H_21=h_21 * y38 * y42
    H_22=h_22 * y41
    H_23=h_23 * y42
    H_24=h_24 * y42
    H_25=h_25 * y37
    H_26=h_26 * y41
    H_27=h_27 * y44
    H_28=h_28 * y44
    H_29=h_29 * y44

    J_0 = j_0 * y7
    J_1 = j_1 * y0
    J_2 = j_2 * y5
    J_3 = j_3 * y3
    J_4 = j_4 * y1
    
    # вычисление вектора F(t) в точке t

    #                                 Метаболиты
    # 1. Adipocyte
    right_y17=2.0*A_7 - A_3
    right_y15=A_1 - A_17 - A_18 - A_19 
    right_y8=A_4 - A_5 - A_6
    right_y9=A_5 + (1.0/2.0)*A_6 + A_9 - A_7 - A_8
    right_y10=A_8 + (1.0/2.0)*A_12 + (1.0/2.0)*A_19 - A_10 - A_11
    right_y11=A_10 + (1.0/2.0)*A_14 + (1.0/2.0)*A_18 - A_13 - A_16
    right_y12=A_2 + 2.0*A_13 - A_7
    right_y13=2.0*A_16 - A_14 - A_15
    right_y14=A_11 + (1.0/2.0)*A_14 + A_15 +(1.0/2.0)*A_17 - A_9 - A_12 - A_16 
    right_y16=(1.0/2.0)*A_6 + (1.0/2.0)*A_12 - A_13
    # 2. Hepatocyte
    right_y34= H_10 - H_11
    right_y35= H_3 + H_11 + H_13 - H_2 - H_10 - H_12 - H_14
    right_y36= H_4 + H_12 + (1.0/2.0)*H_14 + H_23 - H_13 - H_15 - H_20
    right_y37=    H_5 + H_15 + (1.0/2.0)*H_24 + (1.0/2.0)*H_29 - H_16 - H_25
    right_y38= H_16 + H_18 + H_26 + (1.0/2.0)*H_27 - H_17 - H_19 - H_6
    right_y39= H_8 + 2.0*H_19 - H_18 - H_20
    right_y45= 2.0*H_20 - H_9
    right_y40=    H_17 - H_7
    right_y42=    H_22 + H_25 + H_26  + (1.0/2.0)*H_28 - H_21 - H_23 - H_24
    right_y41=    H_21 - H_22 - H_26 
    right_y44= H_1 - H_27 - H_28 - H_29
    right_y43=  (1.0/2.0)*H_14 + (1.0/2.0)*H_24 - H_19
    # 3. Myocyte
    right_y18= M_7 - M_8
    right_y19= M_1 + M_8 - M_7 - M_9
    right_y20= M_9 - M_10
    right_y21=    (1.0/3.0)*M_10 + (1.0/2.0)*M_17 - M_11 - M_2
    right_y22= (1.0/2.0)*M_3 + (1.0/2.0)*M_11 + (1.0/2.0)*M_12 + (1.0/2.0)*M_18 - M_13
    right_y23= M_4 - M_12
    right_y28= M_5 + M_21 - M_6 - M_17 - M_18 - M_19 - M_20
    right_y24=    2.0*M_13 - M_14
    right_y25=    (1.0/2.0)*M_14 + (1.0/2.0)*M_19 - M_13
    right_y26=  (1.0/3.0)*M_10  - M_15 - M_2
    right_y27=  (1.0/2.0)*M_3 + (1.0/2.0)*M_12 + M_15 - M_16
    right_y30=    (1.0/2.0)*M_11 + (1.0/2.0)*M_14
    right_y31=    (1.0/2.0)*M_16
    right_y32=    (1.0/3.0)*M_10 
    right_y33=    (1.0/2.0)*M_16 


    right_y0 = J_carb_flow + H_2 - J_Glu_minus - M_1 - A_4 - H_3   - J_1
    right_y1 =  J_prot_flow + M_6  - J_AA_minus - M_5 - A_1 - H_1  - J_4 
    right_y3= J_0 + (1.0/2.0)*A_3  - J_FFA_minus - M_4 - A_2 - H_8 - J_3  
    right_y5=  - J_KB_minus + J_KB_plus - M_3 - J_2 + H_6

    right_y7 =  J_fat_flow + H_9 - J_0

    right_y2 =    J_0 + (1.0/2.0)*A_3 - H_4
    right_y4=  2.0*M_2 - H_5

    right_y49=    J_4 + (1.0/2.0)*A_17 + (1.0/2.0)*A_18 + (1.0/2.0)*A_19 + (1.0/2.0)*M_17 + (1.0/2.0)*M_18 + (1.0/2.0)*M_19 + (1.0/2.0)*H_27 + (1.0/2.0)*H_28 + (1.0/2.0)*H_29
    right_y6= H_7

    alpha = alpha_base
    beta = beta_base
    gamma = gamma_base
    CL_INS = CL_INS_base
    CL_GLN = CL_GLN_base
    CL_CAM = CL_CAM_base

    # right_INS= alpha * J_carb_flow +beta * J_fat_flow + gamma * J_prot_flow - CL_INS * INS
    # V_extr_fl = 14.0 [L]
    # y0/V_extracerular_fluid [mmol/L]
    # INS [mmol]


    right_y46 =  - y46 * CL_INS  +1.0 * J_carb_flow  +1.0 * J_fat_flow + 1.0 * J_prot_flow  # +1.0 * y0 * Heviside((y0-5.0)/14.0) #
    
    # y0/V_extracerular_fluid [mmol/L]
    # GLN [mmol]
    right_y48 = - CL_GLN * y48  + lambda_ * (1.0/np.maximum(y0/14.0, 0.1)) # не химическая кинетика
    right_y47 = sigma * HeartRate - CL_CAM * y47
    right_y29 = M_20 - M_21
    
    buffer[0] = right_y0
    buffer[1] = right_y1
    buffer[2] = right_y2
    buffer[3] = right_y3
    buffer[4] = right_y4
    buffer[5] = right_y5
    buffer[6] = right_y6
    buffer[7] = right_y7
    buffer[8] = right_y8
    buffer[9] = right_y9
    buffer[10] = right_y10
    buffer[11] = right_y11
    buffer[12] = right_y12
    buffer[13] = right_y13
    buffer[14] = right_y14
    buffer[15] = right_y15
    buffer[16] = right_y16
    buffer[17] = right_y17
    buffer[18] = right_y18
    buffer[19] = right_y19
    buffer[20] = right_y20
    buffer[21] = right_y21
    buffer[22] = right_y22
    buffer[23] = right_y23
    buffer[24] = right_y24
    buffer[25] = right_y25
    buffer[26] = right_y26
    buffer[27] = right_y27
    buffer[28] = right_y28
    buffer[29] = right_y29
    buffer[30] = right_y30
    buffer[31] = right_y31
    buffer[32] = right_y32
    buffer[33] = right_y33
    buffer[34] = right_y34
    buffer[35] = right_y35
    buffer[36] = right_y36
    buffer[37] = right_y37
    buffer[38] = right_y38
    buffer[39] = right_y39
    buffer[40] = right_y40
    buffer[41] = right_y41
    buffer[42] = right_y42
    buffer[43] = right_y43
    buffer[44] = right_y44
    buffer[45] = right_y45
    buffer[46] = right_y46
    buffer[47] = right_y47
    buffer[48] = right_y48
    buffer[49] = right_y49

    return buffer


def J_t(y_vec):
    buffer = np.zeros(shape=(50, ),dtype=np.float32)
    J_carb_flow = 0.0
    J_prot_flow = 0.0
    J_fat_flow  = 0.0
    HeartRate = 80.0

    # Y_{t} values
    # значения в момент времени t
    y0 = y_vec[0]                  
    y1 = y_vec[1]                   
    y2 = y_vec[2]             
    y3 = y_vec[3]                 
    y4 = y_vec[4]                   
    y5 = y_vec[5]                  
    y6  = y_vec[6]           
    y7 = y_vec[7]                   
    y8 = y_vec[8]                    
    y9 = y_vec[9]            
    y10 = y_vec[10]           
    y11 = y_vec[11]        
    y12 = y_vec[12]        
    y13 = y_vec[13]           
    y14 = y_vec[14]           
    y15 = y_vec[15]            
    y16 = y_vec[16]         
    y17 = y_vec[17]                     
    y18 = y_vec[18]                     
    y19 = y_vec[19]            
    y20 = y_vec[20]            
    y21 = y_vec[21]           
    y22 = y_vec[22]        
    y23 = y_vec[23]        
    y24 = y_vec[24]           
    y25 = y_vec[25]           
    y26 = y_vec[26]         
    y27 = y_vec[27]         
    y28 = y_vec[28]            
    y29 = y_vec[29]                 
    y30 = y_vec[30]           
    y31 = y_vec[31]           
    y32 = y_vec[32]        
    y33 = y_vec[33]        
    y34 = y_vec[34]                    
    y35 = y_vec[35]            
    y36 = y_vec[36]            
    y37 = y_vec[37]           
    y38 = y_vec[38]        
    y39 = y_vec[39]        
    y40 = y_vec[40]           
    y41 = y_vec[41]           
    y42 = y_vec[42]           
    y43 = y_vec[43]         
    y44 = y_vec[44]            
    y45 = y_vec[45]
    y46 = y_vec[46]
    y47 = y_vec[47]
    y48 = y_vec[48]            
    y49 = y_vec[49]                 

    is_insulin_process = 1.0
    a_2 = is_insulin_process * a_2_base
    a_4 = is_insulin_process * a_4_base
    a_7 = is_insulin_process * a_7_base
    m_1 = is_insulin_process * m_1_base
    m_7 = is_insulin_process * m_7_base
    h_3 = is_insulin_process * h_3_base
    h_10 = is_insulin_process * h_10_base
    h_19 = is_insulin_process * h_19_base
    h_20 = is_insulin_process * h_20_base

    h_12 = h_12_base
    h_24 = h_24_base
    h_17 = h_17_base
    h_16 = h_16_base
    h_26 = h_26_base
    h_7 = h_7_base
    j_0 = j_0_base
    a_5 = a_5_base
    a_13 = a_13_base
    a_14 = a_14_base
    a_10 = a_10_base
    a_12 = a_12_base
    m_9 = m_9_base
    m_11 = m_11_base


    is_glucagon_adrenalin_insulin_process = 1.0
    h_11 = is_glucagon_adrenalin_insulin_process * h_11_base 
    h_25 = is_glucagon_adrenalin_insulin_process * h_25_base
    h_6 = is_glucagon_adrenalin_insulin_process * h_6_base
    a_3 = is_glucagon_adrenalin_insulin_process * a_3_base
    a_11 = is_glucagon_adrenalin_insulin_process * a_11_base
    m_8 = is_glucagon_adrenalin_insulin_process * m_8_base
    

    is_glucagon_adrenalin_process = 1.0
    h_23 = is_glucagon_adrenalin_process * h_23_base
    h_18 = is_glucagon_adrenalin_process * h_18_base 
    h_13 = is_glucagon_adrenalin_process * h_13_base
    h_2 = is_glucagon_adrenalin_process *  h_2_base
    a_9 = is_glucagon_adrenalin_process *  a_9_base

    a_3 = is_glucagon_adrenalin_process * a_3_base
    m_8 = is_glucagon_adrenalin_process * m_8_base 
    h_12 = is_glucagon_adrenalin_process * h_2_base
    h_11 = is_glucagon_adrenalin_process * h_11_base
    h_13 = is_glucagon_adrenalin_process * h_13_base

    J_AA_minus  = 0.0
    J_Glu_minus  = 0.0
    J_FFA_minus  = 0.0
    J_KB_minus  =  0.0
    J_KB_plus = 0.0

    m_2 = m_2_base 
    m_3 = m_3_base 
    m_4 = m_4_base 
    m_5 = m_5_base 
    m_6 = m_6_base 
    m_10 = m_10_base


    m_12 = m_12_base
    m_13 = m_13_base
    m_14 = m_14_base
    m_15 = m_15_base
    m_16 = m_16_base
    m_17 = m_17_base
    m_18 = m_18_base
    m_19 = m_19_base
    m_20 = m_20_base
    m_21 = m_21_base

    a_1 = a_1_base
    a_6 = a_6_base
    a_8 = a_8_base

    a_15 = a_15_base
    a_16 = a_16_base
    a_17 = a_17_base
    a_18 = a_18_base
    a_19 = a_19_base

    h_1 = h_1_base
    h_4 = h_4_base
    h_5 = h_5_base
    h_8 = h_8_base
    h_9 = h_9_base
    h_14 = h_14_base
    h_15 = h_15_base
    h_21 = h_21_base
    h_22 = h_22_base
    h_27 = h_27_base
    h_28 = h_28_base
    h_29 = h_29_base

    j_1 = j_1_base
    j_2 = j_2_base
    j_3 = j_3_base
    j_4 = j_4_base

    # 2. Myocyte
    M_1 = m_1 * y0
    M_2 = m_2 * y21 * y26
    M_3 = m_3 * y5
    M_4 = m_4 * y3
    M_5 = m_5 * y1
    M_6 = m_6 * y28
    M_7 = m_7 * y19
    M_8 = m_8 * y18
    M_9 = m_9 * y19
    M_10 = m_10 * y20
    M_11 = m_11 * y21
    M_12 = m_12 * y23
    M_13 = m_13 * y22 * y25
    M_14 = m_14 * y24
    M_15 = m_15 * y26
    M_16 = m_16 * y27 # * [O2]
    M_17 = m_17 * y28
    M_18 = m_18 * y28
    M_19 = m_19 * y28
    M_20 = m_20 * y28
    M_21 = m_21 * y29
    #3. Adipocyte
    A_1=a_1 * y1
    A_2=a_2 * y3
    A_3=a_3 * y17
    A_4=a_4 * y0
    A_5=a_5 * y8
    A_6=a_6 * y8
    A_7=a_7 * y9 * y12
    A_8=a_8 * y9
    A_9=a_9 * y14
    A_10=a_10 * y10
    A_11=a_11 * y10
    A_12=a_12 * y14
    A_13=a_13 * y11 * y16
    A_14=a_14 * y13
    A_15=a_15 * y13
    A_16=a_16 * y14 * y11
    A_17=a_17 * y15
    A_18=a_18 * y15
    A_19=a_19 * y15
    #4. Hepatocyte
    H_1=h_1 * y1
    H_2=h_2 * y35
    H_3=h_3 * y0
    H_4=h_4 * y2
    H_5=h_5 * y4
    H_6=h_6 * y38
    H_7=h_7 * y40
    H_8=h_8 * y3
    H_9=h_9 * y45
    H_10=h_10 * y35
    H_11=h_11 * y34
    H_12=h_12 * y35
    H_13=h_13 * y36
    H_14=h_14 * y35
    H_15=h_15 * y36
    H_16=h_16 * y37
    H_17=h_17 * y38 * y43
    H_18=h_18 * y39
    H_19=h_19 * y38 * y43
    H_20=h_20 * y36 * y39
    H_21=h_21 * y38 * y42
    H_22=h_22 * y41
    H_23=h_23 * y42
    H_24=h_24 * y42
    H_25=h_25 * y37
    H_26=h_26 * y41
    H_27=h_27 * y44
    H_28=h_28 * y44
    H_29=h_29 * y44

    J_0 = j_0 * y7
    J_1 = j_1 * y0
    J_2 = j_2 * y5
    J_3 = j_3 * y3
    J_4 = j_4 * y1
    

    J = np.zeros(shape=(len(buffer),len(buffer)),dtype=np.float32)
    # вычисление вектора F(t) в точке t

    #                                 Метаболиты
    # 1. Adipocyte
    right_y17=2.0*A_7 - A_3
    right_y15=A_1 - A_17 - A_18 - A_19 
    right_y8=A_4 - A_5 - A_6
    right_y9=A_5 + (1.0/2.0)*A_6 + A_9 - A_7 - A_8
    right_y10=A_8 + (1.0/2.0)*A_12 + (1.0/2.0)*A_19 - A_10 - A_11
    right_y11=A_10 + (1.0/2.0)*A_14 + (1.0/2.0)*A_18 - A_13 - A_16
    right_y12=A_2 + 2.0*A_13 - A_7
    right_y13=2.0*A_16 - A_14 - A_15
    right_y14=A_11 + (1.0/2.0)*A_14 + A_15 +(1.0/2.0)*A_17 - A_9 - A_12 - A_16 
    right_y16=(1.0/2.0)*A_6 + (1.0/2.0)*A_12 - A_13
    # 2. Hepatocyte
    right_y34= H_10 - H_11
    right_y35= H_3 + H_11 + H_13 - H_2 - H_10 - H_12 - H_14
    right_y36= H_4 + H_12 + (1.0/2.0)*H_14 + H_23 - H_13 - H_15 - H_20
    right_y37=    H_5 + H_15 + (1.0/2.0)*H_24 + (1.0/2.0)*H_29 - H_16 - H_25
    right_y38= H_16 + H_18 + H_26 + (1.0/2.0)*H_27 - H_17 - H_19 - H_6
    right_y39= H_8 + 2.0*H_19 - H_18 - H_20
    right_y45= 2.0*H_20 - H_9
    right_y40=    H_17 - H_7
    right_y42=    H_22 + H_25 + H_26  + (1.0/2.0)*H_28 - H_21 - H_23 - H_24
    right_y41=    H_21 - H_22 - H_26 
    right_y44= H_1 - H_27 - H_28 - H_29
    right_y43=  (1.0/2.0)*H_14 + (1.0/2.0)*H_24 - H_19
    # 3. Myocyte
    right_y18= M_7 - M_8
    right_y19= M_1 + M_8 - M_7 - M_9
    right_y20= M_9 - M_10
    right_y21=    (1.0/3.0)*M_10 + (1.0/2.0)*M_17 - M_11 - M_2
    right_y22= (1.0/2.0)*M_3 + (1.0/2.0)*M_11 + (1.0/2.0)*M_12 + (1.0/2.0)*M_18 - M_13
    right_y23= M_4 - M_12
    right_y28= M_5 + M_21 - M_6 - M_17 - M_18 - M_19 - M_20
    right_y24=    2.0*M_13 - M_14
    right_y25=    (1.0/2.0)*M_14 + (1.0/2.0)*M_19 - M_13
    right_y26=  (1.0/3.0)*M_10  - M_15 - M_2
    right_y27=  (1.0/2.0)*M_3 + (1.0/2.0)*M_12 + M_15 - M_16
    right_y30=    (1.0/2.0)*M_11 + (1.0/2.0)*M_14
    right_y31=    (1.0/2.0)*M_16
    right_y32=    (1.0/3.0)*M_10 
    right_y33=    (1.0/2.0)*M_16 


    right_y0 = J_carb_flow + H_2 - J_Glu_minus - M_1 - A_4 - H_3   - J_1
    right_y1 =  J_prot_flow + M_6  - J_AA_minus - M_5 - A_1 - H_1  - J_4 
    right_y3= J_0 + (1.0/2.0)*A_3  - J_FFA_minus - M_4 - A_2 - H_8 - J_3  
    right_y5=  - J_KB_minus + J_KB_plus - M_3 - J_2 + H_6

    right_y7 =  J_fat_flow + H_9 - J_0

    right_y2 =    J_0 + (1.0/2.0)*A_3 - H_4
    right_y4=  2.0*M_2 - H_5

    right_y49=    J_4 + (1.0/2.0)*A_17 + (1.0/2.0)*A_18 + (1.0/2.0)*A_19 + (1.0/2.0)*M_17 + (1.0/2.0)*M_18 + (1.0/2.0)*M_19 + (1.0/2.0)*H_27 + (1.0/2.0)*H_28 + (1.0/2.0)*H_29
    right_y6= H_7

    alpha = alpha_base
    beta = beta_base
    gamma = gamma_base
    CL_INS = CL_INS_base
    CL_GLN = CL_GLN_base
    CL_CAM = CL_CAM_base

    right_y46 =  - y46 * CL_INS  +1.0 * J_carb_flow  +1.0 * J_fat_flow + 1.0 * J_prot_flow  # +1.0 * y0 * Heviside((y0-5.0)/14.0) #
    
    # y0/V_extracerular_fluid [mmol/L]
    # GLN [mmol]
    right_y48 = - CL_GLN * y48  + lambda_ * (1.0/np.maximum(y0/14.0, 0.1)) # не химическая кинетика
    right_y47 = sigma * HeartRate - CL_CAM * y47
    right_y29 = M_20 - M_21
    
    buffer[0] = right_y0
    buffer[1] = right_y1
    buffer[2] = right_y2
    buffer[3] = right_y3
    buffer[4] = right_y4
    buffer[5] = right_y5
    buffer[6] = right_y6
    buffer[7] = right_y7
    buffer[8] = right_y8
    buffer[9] = right_y9
    buffer[10] = right_y10
    buffer[11] = right_y11
    buffer[12] = right_y12
    buffer[13] = right_y13
    buffer[14] = right_y14
    buffer[15] = right_y15
    buffer[16] = right_y16
    buffer[17] = right_y17
    buffer[18] = right_y18
    buffer[19] = right_y19
    buffer[20] = right_y20
    buffer[21] = right_y21
    buffer[22] = right_y22
    buffer[23] = right_y23
    buffer[24] = right_y24
    buffer[25] = right_y25
    buffer[26] = right_y26
    buffer[27] = right_y27
    buffer[28] = right_y28
    buffer[29] = right_y29
    buffer[30] = right_y30
    buffer[31] = right_y31
    buffer[32] = right_y32
    buffer[33] = right_y33
    buffer[34] = right_y34
    buffer[35] = right_y35
    buffer[36] = right_y36
    buffer[37] = right_y37
    buffer[38] = right_y38
    buffer[39] = right_y39
    buffer[40] = right_y40
    buffer[41] = right_y41
    buffer[42] = right_y42
    buffer[43] = right_y43
    buffer[44] = right_y44
    buffer[45] = right_y45
    buffer[46] = right_y46
    buffer[47] = right_y47
    buffer[48] = right_y48
    buffer[49] = right_y49

    return buffer