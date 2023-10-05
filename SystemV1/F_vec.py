import numpy as np
import numba 
from input import *
import torch
from local_contributor_config import problem_folder

W =  20.0 # [min] window to check AUC(INS, t-W,t)
INS_check_coeff = 400 # [mmol*s]

# take grid on from FPC.ipynb file 
tau_grid = 0.001 # [min]
t_0 = 400.0 # [min]
t_end = 2500.0 # [min]
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

# номера коэффициентов
a_1_base=            1.0/10.0#10.0**(power_of_coeff)
a_2_base=            1.0/10.0#10.0**(power_of_coeff)
a_3_base=            1.0/10.0#10.0**(power_of_coeff)
a_4_base=            1.0/10.0#10.0**(power_of_coeff)
a_5_base=            1.0/10.0#10.0**(power_of_coeff)
a_6_base=            1.0/10.0#10.0**(power_of_coeff)
a_7_base=            1.0/10.0#10.0**(power_of_coeff)
a_8_base=            1.0/10.0#10.0**(power_of_coeff)
a_9_base=            1.0/10.0#10.0**(power_of_coeff)
a_10_base=            1.0/10.0#10.0**(power_of_coeff)
a_11_base=            1.0/10.0#10.0**(power_of_coeff)
a_12_base=            1.0/10.0#10.0**(power_of_coeff)
a_13_base=            1.0/10.0#10.0**(power_of_coeff)
a_14_base=            1.0/10.0#10.0**(power_of_coeff)
a_15_base=            1.0/10.0#10.0**(power_of_coeff)
a_16_base=            1.0/10.0#10.0**(power_of_coeff)
a_17_base=            1.0/10.0#10.0**(power_of_coeff)
a_18_base=            1.0/10.0#10.0**(power_of_coeff)
a_19_base=            1.0/10.0#10.0**(power_of_coeff)
m_1_base=            1.0/10.0#10.0**(power_of_coeff)
m_2_base=            1.0/10.0#10.0**(power_of_coeff)
m_3_base=            1.0/10.0#10.0**(power_of_coeff)
m_4_base=            1.0/10.0#10.0**(power_of_coeff)
m_5_base=            1.0/10.0#10.0**(power_of_coeff)
m_6_base=            1.0/10.0#10.0**(power_of_coeff)
m_7_base=            1.0/10.0#10.0**(power_of_coeff)
m_8_base=            1.0/10.0#10.0**(power_of_coeff)
m_9_base=            1.0/10.0#10.0**(power_of_coeff)
m_10_base=           1.0/10.0#10.0**(power_of_coeff)
m_11_base=           1.0/10.0#10.0**(power_of_coeff)
m_12_base=           1.0/10.0#10.0**(power_of_coeff)  # *[Carnitin]
m_13_base=           1.0/10.0#10.0**(power_of_coeff)
m_14_base=           1.0/10.0#10.0**(power_of_coeff)
m_15_base=           1.0/10.0#10.0**(power_of_coeff)
m_16_base=           1.0/10.0#10.0**(power_of_coeff) # *[Creatin]
m_17_base=           1.0/10.0#10.0**(power_of_coeff)
m_18_base=           1.0/10.0#10.0**(power_of_coeff)
m_19_base=           1.0/10.0#10.0**(power_of_coeff)
m_20_base=           1.0/10.0#10.0**(power_of_coeff)
m_21_base=           1.0/10.0#10.0**(power_of_coeff)
h_1_base=            1.0/10.0#10.0**(power_of_coeff)
h_2_base=            1.0/10.0#10.0**(power_of_coeff)
h_3_base=            1.0/10.0#10.0**(power_of_coeff)
h_4_base=            1.0/10.0#10.0**(power_of_coeff)
h_5_base=            1.0/10.0#10.0**(power_of_coeff)
h_6_base=            1.0/10.0#10.0**(power_of_coeff)
h_7_base=            1.0/10.0#10.0**(power_of_coeff)
h_8_base=            1.0/10.0#10.0**(power_of_coeff)
h_9_base=            1.0/10.0#10.0**(power_of_coeff)
h_10_base=            1.0/10.0#10.0**(power_of_coeff)
h_11_base=            1.0/10.0#10.0**(power_of_coeff)
h_12_base=            1.0/10.0#10.0**(power_of_coeff)
h_13_base=            1.0/10.0#10.0**(power_of_coeff)
h_14_base=            1.0/10.0#10.0**(power_of_coeff)
h_15_base=            1.0/10.0#10.0**(power_of_coeff)
h_16_base=            1.0/10.0#10.0**(power_of_coeff)
h_17_base=            1.0/10.0#10.0**(power_of_coeff)
h_18_base=            1.0/10.0#10.0**(power_of_coeff)
h_19_base=            1.0/10.0#10.0**(power_of_coeff)
h_20_base=            1.0/10.0#10.0**(power_of_coeff)
h_21_base=            1.0/10.0#10.0**(power_of_coeff)
h_22_base=            1.0/10.0#10.0**(power_of_coeff)
h_23_base=            1.0/10.0#10.0**(power_of_coeff)
h_24_base=            1.0/10.0#10.0**(power_of_coeff)
h_25_base=            1.0/10.0#10.0**(power_of_coeff)
h_26_base=            1.0/10.0#10.0**(power_of_coeff)
h_27_base=            1.0/10.0#10.0**(power_of_coeff)
h_28_base=            1.0/10.0#10.0**(power_of_coeff)
h_29_base=            1.0/10.0#10.0**(power_of_coeff)

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

# @jit(nopython = True)
def F_vec(t: float, y_vec: np.array,
          INS_on_grid:np.array, INS_AUC_w_on_grid:np.array,T_a_on_grid:np.array,
          last_seen_time:np.array,last_time_pos:np.array,
            J_flow_carb_vs:np.array,
            J_flow_prot_vs:np.array,
            J_flow_fat_vs:np.array):
    buffer = np.zeros(shape=(50,))
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
    Glu_ef = y_vec[0]                  
    AA_ef = y_vec[1]                   
    Glycerol_ef = y_vec[2]             
    FFA_ef = y_vec[3]                 
    Lac_m = y_vec[4]                   
    KB_ef = y_vec[5]                  
    Cholesterol_pl   = y_vec[6]           
    TG_pl = y_vec[7]                   
    G6_a = y_vec[8]                    
    G3_a = y_vec[9]            
    Pyr_a = y_vec[10]           
    Ac_CoA_a = y_vec[11]        
    FA_CoA_a = y_vec[12]        
    Cit_a = y_vec[13]           
    OAA_a = y_vec[14]           
    AA_a = y_vec[15]            
    NADPH_a = y_vec[16]         
    TG_a = y_vec[17]                     
    GG_m = y_vec[18]                     
    G6_m = y_vec[19]            
    G3_m = y_vec[20]            
    Pyr_m = y_vec[21]           
    Ac_CoA_m = y_vec[22]        
    FA_CoA_m = y_vec[23]        
    Cit_m = y_vec[24]           
    OAA_m = y_vec[25]           
    H_cyt_m = y_vec[26]         
    H_mit_m = y_vec[27]         
    AA_m = y_vec[28]            
    Muscle_m = y_vec[29]                 
    CO2_m = y_vec[30]           
    H2O_m = y_vec[31]           
    ATP_cyt_m = y_vec[32]        
    ATP_mit_m = y_vec[33]        
    GG_h = y_vec[34]                    
    G6_h = y_vec[35]            
    G3_h = y_vec[36]            
    Pyr_h = y_vec[37]           
    Ac_CoA_h = y_vec[38]        
    FA_CoA_h = y_vec[39]        
    MVA_h = y_vec[40]           
    Cit_h = y_vec[41]           
    OAA_h = y_vec[42]           
    NADPH_h = y_vec[43]         
    AA_h = y_vec[44]            
    TG_h = y_vec[45]
    INS = y_vec[46]
    CAM = y_vec[47]
    GLN = y_vec[48]            
    Urea_ef = y_vec[49]                 


    insulin_activation_coefficient =  17.0
    is_insulin_process = Heviside(INS-insulin_activation_coefficient)
    # is_insulin_process = Sigmoid(INS-insulin_activation_coefficient)
    h_10 = is_insulin_process * h_10_base 
    h_12 = is_insulin_process * h_12_base
    h_24 = is_insulin_process * h_24_base
    h_20 = is_insulin_process * h_20_base
    h_17 = is_insulin_process * h_17_base
    h_16 = is_insulin_process * h_16_base
    h_26 = is_insulin_process * h_26_base
    h_19 = is_insulin_process * h_19_base
    h_7 = is_insulin_process * h_7_base
    h_3 = is_insulin_process * h_3_base
    j_0 = is_insulin_process * j_0_base
    a_4 = is_insulin_process * a_4_base
    a_2 = is_insulin_process * a_2_base
    a_5 = is_insulin_process * a_5_base
    a_7 = is_insulin_process * a_7_base
    a_13 = is_insulin_process * a_13_base
    a_14 = is_insulin_process * a_14_base
    a_10 = is_insulin_process * a_10_base
    a_12 = is_insulin_process * a_12_base
    m_7 = is_insulin_process * m_7_base
    m_9 = is_insulin_process * m_9_base
    m_1 = is_insulin_process * m_1_base
    m_11 = is_insulin_process * m_11_base

    # glucagon_adrenilin_activation_coefficient = GLN+CAM
    # is_glucagon_adrenalin_process = Heviside(glucagon_adrenilin_activation_coefficient-160.0)
    # is_glucagon_adrenalin_process = Sigmoid(glucagon_adrenilin_activation_coefficient-160.0)
    # h_23 = is_glucagon_adrenalin_process * h_23_base
    # h_18 = is_glucagon_adrenalin_process * h_18_base 
    # h_13 = is_glucagon_adrenalin_process * h_13_base
    # h_2 = is_glucagon_adrenalin_process *  h_2_base
    # a_9 = is_glucagon_adrenalin_process *  a_9_base


    glucagon_adrenalin_insulin_activation_coefficient = INS/(GLN+CAM)
    is_glucagon_adrenalin_insulin_process = Heviside(glucagon_adrenalin_insulin_activation_coefficient-1.0)
    # is_glucagon_adrenalin_insulin_process = Sigmoid(glucagon_adrenalin_insulin_activation_coefficient-1.0)
    h_11 = is_glucagon_adrenalin_insulin_process * h_11_base 
    h_25 = is_glucagon_adrenalin_insulin_process * h_25_base
    h_6 = is_glucagon_adrenalin_insulin_process * h_6_base
    a_3 = is_glucagon_adrenalin_insulin_process * a_3_base
    a_11 = is_glucagon_adrenalin_insulin_process * a_11_base
    m_8 = is_glucagon_adrenalin_insulin_process * m_8_base

    is_glucagon_adrenalin_process = 1.0 - is_glucagon_adrenalin_insulin_process
    h_23 = is_glucagon_adrenalin_process * h_23_base
    h_18 = is_glucagon_adrenalin_process * h_18_base 
    h_13 = is_glucagon_adrenalin_process * h_13_base
    h_2 = is_glucagon_adrenalin_process *  h_2_base
    a_9 = is_glucagon_adrenalin_process *  a_9_base

    # if not int(is_glucagon_adrenalin_insulin_process):


    AUC_at_t = -1.0
    T_a_t = -1.0

    # print(last_time_pos[0],t_pos)
    if t_pos - last_time_pos[0] > 0:
        diff_ = t_pos - last_time_pos[0]
        T_a_current = T_a_on_grid[last_time_pos[0]]
        last_seen_time[0] = t 
        t_minus_w_pos = np.maximum(np.intc(0), np.intc((t-W-t_0)/tau_grid))

        for j in range(1,diff_+1):
            INS_on_grid[last_time_pos[0]+j] = INS

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
    M_1 = m_1 * Glu_ef
    M_2 = m_2 * Pyr_m
    M_3 = m_3 * KB_ef
    M_4 = m_4 * FFA_ef
    M_5 = m_5 * AA_ef
    M_6 = m_6 * AA_m
    M_7 = m_7 * G6_m
    M_8 = m_8 * GG_m
    M_9 = m_9 * G6_m
    M_10 = m_10 * G3_m
    M_11 = m_11 * Pyr_m
    M_12 = m_12 * FA_CoA_m
    M_13 = m_13 * Ac_CoA_m * OAA_m
    M_14 = m_14 * Cit_m
    M_15 = m_15 * H_cyt_m
    M_16 = m_16 * H_mit_m # * [O2]
    M_17 = m_17 * AA_m
    M_18 = m_18 * AA_m
    M_19 = m_19 * AA_m
    M_20 = m_20 * AA_m
    M_21 = m_21 * Muscle_m
    #3. Adipocyte
    A_1=a_1 * AA_ef
    A_2=a_2 * FFA_ef
    A_3=a_3 * TG_a
    A_4=a_4 * Glu_ef
    A_5=a_5 * G6_a
    A_6=a_6 * G6_a
    A_7=a_7 * G3_a * FA_CoA_a
    A_8=a_8 * G3_a
    A_9=a_9 * OAA_a
    A_10=a_10 * Pyr_a
    A_11=a_11 * Pyr_a
    A_12=a_12 * OAA_a
    A_13=a_13 * Ac_CoA_a * NADPH_a
    A_14=a_14 * Cit_a
    A_15=a_15 * Cit_a
    A_16=a_16 * OAA_a * Cit_a
    A_17=a_17 * AA_a
    A_18=a_18 * AA_a
    A_19=a_19 * AA_a
    #4. Hepatocyte
    H_1=h_1 * AA_ef
    H_2=h_2 * G6_h
    H_3=h_3 * Glu_ef
    H_4=h_4 * Glycerol_ef
    H_5=h_5 * Lac_m
    H_6=h_6 * Ac_CoA_h
    H_7=h_7 * MVA_h
    H_8=h_8 * FFA_ef
    H_9=h_9 * TG_h
    H_10=h_10 * G6_h
    H_11=h_11 * GG_h
    H_12=h_12 * G6_h
    H_13=h_13 * G3_h
    H_14=h_14 * G6_h
    H_15=h_15 * G3_h
    H_16=h_16 * Pyr_h
    H_17=h_17 * Ac_CoA_h * NADPH_h
    H_18=h_18 * FA_CoA_h
    H_19=h_19 * Ac_CoA_h * NADPH_h
    H_20=h_20 * G3_h * FA_CoA_h
    H_21=h_21 * Ac_CoA_h * OAA_h
    H_22=h_22 * Cit_h
    H_23=h_23 * OAA_h
    H_24=h_24 * OAA_h
    H_25=h_25 * Pyr_h
    H_26=h_26 * AA_h
    H_27=h_27 * AA_h
    H_28=h_28 * AA_h
    H_29=h_29 * AA_h

    J_0 = j_0 * TG_pl
    J_1 = j_1 * Glu_ef
    J_2 = j_2 * KB_ef
    J_3 = j_3 * FFA_ef
    J_4 = j_4 * AA_ef

    
    # BMR
    e_AA_min = 0.1*e_sigma
    e_Glu_min = 0.2*e_sigma
    e_FFA_min = 0.035*e_sigma
    e_KB_min = 0.0

    e_AA_minus = 0.0
    e_Glu_minus = 0.0
    e_FFA_minus = 0.0
    e_KB_minus = 0.0

    if AA_ef >= 20.0:
        e_AA_minus = e_sigma - (e_Glu_min+e_KB_min+e_FFA_min)
        e_Glu_minus = e_Glu_min
        e_FFA_minus = e_FFA_min
        e_KB_minus = e_KB_min
    elif (AA_ef < 20.0) and (Glu_ef>=20.0) and (T_a_t==0.0):
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



    e_KB_plus = 0.005*e_sigma
    J_KB_plus = e_KB_plus*inv_beta_KB*Heviside(T_a_t-180.0)
    J_AA_minus  = e_AA_minus*inv_beta_AA*Heviside(AA_ef-tau_grid*e_AA_minus*inv_beta_AA)
    J_Glu_minus  = e_Glu_minus*inv_beta_Glu*Heviside(Glu_ef-tau_grid*e_Glu_minus*inv_beta_Glu)
    J_FFA_minus  = e_FFA_minus*inv_beta_AA*Heviside(FFA_ef-tau_grid*e_FFA_minus*inv_beta_FFA)
    J_KB_minus  =  e_KB_minus*inv_beta_KB*Heviside(KB_ef-tau_grid*e_KB_minus*inv_beta_KB)

    
    # J_AA_minus  = 0.0
    # J_Glu_minus  = 0.0
    # J_FFA_minus  = 0.0
    # J_KB_minus  =  0.0
    # J_KB_plus = 0.0
    



    # непостредственно вычисление вектора F(t) в точке t

    #                                 Метаболиты
    # 1. Adipocyte
    right_TG_a=A_7 - A_3
    right_AA_a=A_1 - A_17 - A_18 - A_19 
    right_G6_a=A_4 - A_5 - A_6
    right_G3_a=2*A_5 + A_6 + A_9 - A_7 - A_8
    right_Pyr_a=A_8 + A_12 + A_19 - A_10 - A_11
    right_Ac_CoA_a=A_10 + A_14 + A_18 - 8*A_13 - A_16
    right_FA_CoA_a=A_2 + A_13 - 3*A_7 # 8*A_13
    right_Cit_a=A_16 - A_14 - A_15
    right_OAA_a=A_11 + A_14 + A_15 + A_17 - A_9 - A_12 - A_16 
    right_NADPH_a=A_6 + A_12 - 14*A_13
    # 2. Hepatocyte
    right_GG_h= H_10 - H_11
    right_G6_h= H_3 + H_11 + H_13 - H_2 - H_10 - H_12 - H_14
    right_G3_h= H_4 + 2*H_12 + H_14 + H_23 - 2*H_13 - H_15 - H_20
    right_Pyr_h=    H_5 + H_15 + H_24 + H_29 - H_16 - H_25
    right_Ac_CoA_h= H_16 + 8*H_18 + H_26 + H_27 - 3*H_17 - 8*H_19
    right_FA_CoA_h= H_8 + H_19 - H_18 - 3*H_20
    right_TG_h= H_20 - H_9
    right_MVA_h=    H_17 - H_7
    right_OAA_h=    H_22 + H_25 + H_26  + H_28 - H_21 - H_23 - H_24
    right_Cit_h=    H_21 - H_22 - H_26
    right_AA_h= H_1 - H_27 - H_28 - H_29
    right_NADPH_h=  6*H_14 + H_24 - 14*H_19
    # 3. Myocyte
    right_GG_m= M_7 - M_8
    right_G6_m= M_1 + M_8 - M_7 - M_9
    right_G3_m= 2*M_9 - M_10
    right_Pyr_m=    M_10 + M_17 - M_11 - M_2
    right_Ac_CoA_m= 2*M_3 + M_11 + 8*M_12 + M_18 - M_13
    right_FA_CoA_m= M_4 - M_12
    right_AA_m= M_5 + M_21 - M_6 - M_17 - M_18 - M_19 - M_20
    right_Cit_m=    M_13 - M_14
    right_OAA_m=    M_14 + M_19 - M_13
    right_H_cyt_m=  2*M_10 - M_2 - M_15
    right_H_mit_m=  M_3 + 14*M_12 + M_15 - M_16
    right_CO2_m=    M_11 + 2*M_14
    right_H2O_m=    M_16
    right_ATP_cyt_m=    M_10 # Anaerob
    right_ATP_mit_m=    2*M_16 # Aerob


    right_Glu_ef = J_carb_flow + H_2 - J_Glu_minus - H_3 - M_1 - A_4 - J_1
    right_AA_ef =  J_prot_flow + M_6  - J_AA_minus - A_1 - H_1 - M_5 - J_4 
    right_FFA_ef= 3*J_0 + 3*A_3  - J_FFA_minus - J_3 - A_2 - H_8 - M_4
    right_KB_ef=  H_6  - J_KB_minus + J_KB_plus - J_2 - M_3

    right_TG_pl =  J_fat_flow + H_9 - J_0 

    right_Glycerol_ef =    J_0 + A_3 - H_4
    right_Lac_m=  M_2 - H_5

    right_Urea_ef=    J_4 + A_17 + A_18 + A_19 + M_17 + M_18 + M_19 + H_27 + H_28 + H_29
    right_Cholesterol_pl= H_7

    alpha = alpha_base
    beta = beta_base
    gamma = gamma_base
    CL_INS = CL_INS_base
    CL_GLN = CL_GLN_base
    CL_CAM = CL_CAM_base

    # right_INS= alpha * J_carb_flow +beta * J_fat_flow + gamma * J_prot_flow - CL_INS * INS
    # V_extr_fl = 14.0 [L]
    # Glu_ef/V_extracerular_fluid [mmol/L]
    # INS [mmol]


    right_INS= 1.0 * J_carb_flow + 1.0 * Glu_ef * Heviside((Glu_ef-5.0)/14.0) +1.0 * J_fat_flow + 1.0 * J_prot_flow - INS * CL_INS
     
    # Glu_ef/V_extracerular_fluid [mmol/L]
    # GLN [mmol]
    right_GLN = lambda_ * (1.0/np.maximum(Glu_ef/14.0, 0.1)) - CL_GLN * GLN
    right_CAM = sigma * HeartRate - CL_CAM * CAM
    right_Muscle_m = M_20 - M_21



    # output buffer
    # buffer[0] = right_Glu_ef
    # buffer[1] = right_AA_ef
    # buffer[2] = np.maximum(right_Glycerol_ef ,0.0)
    # buffer[3] = right_FFA_ef
    # buffer[4] = np.maximum(right_Lac_m ,0.0)
    # buffer[5] = right_KB_ef
    # buffer[6] = np.maximum(right_Cholesterol_pl ,0.0)
    # buffer[7] = np.maximum(right_TG_pl ,0.0)
    # buffer[8] = np.maximum(right_G6_a ,0.0)
    # buffer[9] = np.maximum(right_G3_a ,0.0)
    # buffer[10] = np.maximum(right_Pyr_a ,0.0)
    # buffer[11] = np.maximum(right_Ac_CoA_a ,0.0)
    # buffer[12] = np.maximum(right_FA_CoA_a ,0.0)
    # buffer[13] = np.maximum(right_Cit_a ,0.0)
    # buffer[14] = np.maximum(right_OAA_a ,0.0)
    # buffer[15] = np.maximum(right_AA_a ,0.0)
    # buffer[16] = np.maximum(right_NADPH_a ,0.0)
    # buffer[17] = np.maximum(right_TG_a ,0.0)
    # buffer[18] = np.maximum(right_GG_m ,0.0)
    # buffer[19] = np.maximum(right_G6_m ,0.0)
    # buffer[20] = np.maximum(right_G3_m ,0.0)
    # buffer[21] = np.maximum(right_Pyr_m ,0.0)
    # buffer[22] = np.maximum(right_Ac_CoA_m ,0.0)
    # buffer[23] = np.maximum(right_FA_CoA_m ,0.0)
    # buffer[24] = np.maximum(right_Cit_m ,0.0)
    # buffer[25] = np.maximum(right_OAA_m ,0.0)
    # buffer[26] = np.maximum(right_H_cyt_m ,0.0)
    # buffer[27] = np.maximum(right_H_mit_m ,0.0)
    # buffer[28] = np.maximum(right_AA_m ,0.0)
    # buffer[29] = np.maximum(right_Muscle_m ,0.0)
    # buffer[30] = np.maximum(right_CO2_m ,0.0)
    # buffer[31] = np.maximum(right_H2O_m ,0.0)
    # buffer[32] = np.maximum(right_ATP_cyt_m ,0.0)
    # buffer[33] = np.maximum(right_ATP_mit_m ,0.0)
    # buffer[34] = np.maximum(right_GG_h ,0.0)
    # buffer[35] = np.maximum(right_G6_h ,0.0)
    # buffer[36] = np.maximum(right_G3_h ,0.0)
    # buffer[37] = np.maximum(right_Pyr_h ,0.0)
    # buffer[38] = np.maximum(right_Ac_CoA_h ,0.0)
    # buffer[39] = np.maximum(right_FA_CoA_h ,0.0)
    # buffer[40] = np.maximum(right_MVA_h ,0.0)
    # buffer[41] = np.maximum(right_Cit_h ,0.0)
    # buffer[42] = np.maximum(right_OAA_h ,0.0)
    # buffer[43] = np.maximum(right_NADPH_h ,0.0)
    # buffer[44] = np.maximum(right_AA_h ,0.0)
    # buffer[45] = np.maximum(right_TG_h ,0.0)
    # buffer[46] = np.maximum(right_INS ,0.0)
    # buffer[47] = np.maximum(right_CAM ,0.0)
    # buffer[48] = np.maximum(right_GLN ,0.0)
    # buffer[49] = np.maximum(right_Urea_ef ,0.0)
    buffer[0] = right_Glu_ef
    buffer[1] = right_AA_ef
    buffer[2] = right_Glycerol_ef
    buffer[3] = right_FFA_ef
    buffer[4] = right_Lac_m
    buffer[5] = right_KB_ef
    buffer[6] = right_Cholesterol_pl
    buffer[7] = right_TG_pl
    buffer[8] = right_G6_a
    buffer[9] = right_G3_a
    buffer[10] = right_Pyr_a
    buffer[11] = right_Ac_CoA_a
    buffer[12] = right_FA_CoA_a
    buffer[13] = right_Cit_a
    buffer[14] = right_OAA_a
    buffer[15] = right_AA_a
    buffer[16] = right_NADPH_a
    buffer[17] = right_TG_a
    buffer[18] = right_GG_m
    buffer[19] = right_G6_m
    buffer[20] = right_G3_m
    buffer[21] = right_Pyr_m
    buffer[22] = right_Ac_CoA_m
    buffer[23] = right_FA_CoA_m
    buffer[24] = right_Cit_m
    buffer[25] = right_OAA_m
    buffer[26] = right_H_cyt_m
    buffer[27] = right_H_mit_m
    buffer[28] = right_AA_m
    buffer[29] = right_Muscle_m
    buffer[30] = right_CO2_m
    buffer[31] = right_H2O_m
    buffer[32] = right_ATP_cyt_m
    buffer[33] = right_ATP_mit_m
    buffer[34] = right_GG_h
    buffer[35] = right_G6_h
    buffer[36] = right_G3_h
    buffer[37] = right_Pyr_h
    buffer[38] = right_Ac_CoA_h
    buffer[39] = right_FA_CoA_h
    buffer[40] = right_MVA_h
    buffer[41] = right_Cit_h
    buffer[42] = right_OAA_h
    buffer[43] = right_NADPH_h
    buffer[44] = right_AA_h
    buffer[45] = right_TG_h
    buffer[46] = right_INS
    buffer[47] = right_CAM
    buffer[48] = right_GLN
    buffer[49] = right_Urea_ef
    if len(np.argwhere(np.abs(buffer) > 10**5)) !=0:
        where_bad = np.argwhere(np.abs(buffer) > 10**5)
        print(t)

    # if len(np.argwhere(np.isnan(buffer))) !=0 :
    #     print(t)
    return buffer
