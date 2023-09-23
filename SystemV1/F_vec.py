import numpy as np
import numba 
from sysv1_de_config import F_carb
from sysv1_de_config import F_fat
from sysv1_de_config import F_prot
from sysv1_de_config import sigmoid
from input import *
import torch
from local_contributor_config import problem_folder

# take grid on from FPC.ipynb file 
tau_grid = 0.1 # [min]
t_0 = 0.0 # [min]
t_end = 6000.0 # [min]


# make input data
J_prot_func = torch.load(
    os.path.join(problem_folder, 'ddt_AA_ef'))

J_carb_func = torch.load(
    os.path.join(problem_folder, 'ddt_Glu_ef'))
J_fat_func = torch.load(
    os.path.join(problem_folder, 'ddt_TG_pl'))


lambda_ = 1.0
sigma = 1.0

### INS

CL_INS    = 0.1
alpha       = 2.0
alpha_a2    = 1.0
alpha_a4    = 1.0
alpha_a5    = 1.0
alpha_a7    = 1.0
alpha_a10   =1.0
alpha_m1    = 1.0
alpha_m3    = 1.0
alpha_m6    = 1.0
alpha_m8    = 1.0
alpha_m10   =1.0
alpha_h3    = 1.0
alpha_h6    = 1.0
alpha_h8    = 1.0
alpha_h10   =1.0
alpha_h12   =1.0
alpha_h17   =1.0
alpha_h29   =1.0
alpha_h30   =1.0
beta=         0.02
beta_a2=        2.0
beta_a4=        2.0
beta_a5=        2.0
beta_a7=        2.0
beta_a10=       2.0
beta_m1=        2.0
beta_m3=        2.0
beta_m6=        2.0
beta_m8=        2.0
beta_m10=       2.0
beta_h3=        2.0
beta_h6=        2.0
beta_h8=        2.0
beta_h10=       2.0
beta_h12=       2.0
beta_h17=       2.0
beta_h29=       2.0
beta_h30=       2.0
delta_a2=       2.0
delta_a4=       2.0
delta_a5=       2.0
delta_a7=       2.0
delta_a10=      2.0
delta_m1=       2.0
delta_m3=       2.0
delta_m6=       2.0
delta_m8=       2.0
delta_m10=      2.0
delta_h3=       2.0
delta_h6=       2.0
delta_h8=       2.0
delta_h10=      2.0
delta_h12=      2.0
delta_h17=      2.0
delta_h29=      2.0
delta_h30=      2.0
gamma =        1.0
gamma_a4=       1.0
gamma_a5=       1.0
gamma_a7=       1.0
gamma_a10=      1.0
gamma_m1=       1.0
gamma_m3=       1.0
gamma_m6=       1.0
gamma_m8=       1.0
gamma_m10=      1.0
gamma_h3=       1.0
gamma_h6=       1.0
gamma_h8=       1.0
gamma_h10=      1.0
gamma_h12=      1.0
gamma_h17=      1.0
gamma_h29=      1.0
gamma_h30=      1.0
### GLN
CL_GLN=           0.1
lambda_a3=      1.0
lambda_a9=      1.0
lambda_a11=     1.0
lambda_m7=      1.0
lambda_h2=      1.0
lambda_h11=     1.0
lambda_h15=     1.0
lambda_h25=     1.0
### CAM
CL_CAM=           0.1
sigma_a3=       1.0
sigma_m7=       1.0
sigma_h11=      1.0
tau_carb=     60.0     # [min]
tau_fat=      110.0    # [min]
tau_prot=     90.0     # [min}
a_carb=         1.0
a_fat=          1.0
a_prot  =       1.0
    # номера коэффициентов
a_1=            10.0**(-5)
a_2=            10.0**(-5)
a_3=            10.0**(-5)
a_4=            10.0**(-5)
a_5=            10.0**(-5)
a_6=            10.0**(-5)
a_7=            10.0**(-5)
a_8=            10.0**(-5)
a_9=            10.0**(-5)
a_10=            10.0**(-5)
a_11=            10.0**(-5)
a_12=            10.0**(-5)
a_13=            10.0**(-5)
a_14=            10.0**(-5)
a_15=            10.0**(-5)
a_16=            10.0**(-5)
a_17=            10.0**(-5)
a_18=            10.0**(-5)
a_19=            10.0**(-5)
m_1=            10.0**(-5)
m_2=            10.0**(-5)
m_3=            10.0**(-5)
m_4=            10.0**(-5)
m_5=            10.0**(-5)
m_6=            10.0**(-5)
m_7=            10.0**(-5)
m_8=            10.0**(-5)
m_9=            10.0**(-5)
m_10=           10.0**(-5)
m_11=           10.0**(-5)
m_12=           10.0**(-5)  # *[Carnitin]
m_13=           10.0**(-5)
m_14=           10.0**(-5)
m_15=           10.0**(-5)
m_16=           10.0**(-5) # *[Creatin]
m_17=           10.0**(-5)
m_18=           10.0**(-5)
m_19=           10.0**(-5)
m_20=           10.0**(-5)
m_21=           10.0**(-5)
h_1=            10.0**(-5)
h_2=            10.0**(-5)
h_3=            10.0**(-5)
h_4=            10.0**(-5)
h_5=            10.0**(-5)
h_6=            10.0**(-5)
h_7=            10.0**(-5)
h_8=            10.0**(-5)
h_9=            10.0**(-5)
h_10=            10.0**(-5)
h_11=            10.0**(-5)
h_12=            10.0**(-5)
h_13=            10.0**(-5)
h_14=            10.0**(-5)
h_15=            10.0**(-5)
h_16=            10.0**(-5)
h_17=            10.0**(-5)
h_18=            10.0**(-5)
h_19=            10.0**(-5)
h_20=            10.0**(-5)
h_21=            10.0**(-5)
h_22=            10.0**(-5)
h_23=            10.0**(-5)
h_24=            10.0**(-5)
h_25=            10.0**(-5)
h_26=            10.0**(-5)
h_27=            10.0**(-5)
h_28=            10.0**(-5)
h_29=            10.0**(-5)


j_0 = 1.0
j_1 = 1.0
j_2 = 1.0
j_3 = 1.0
j_4 = 1.0

# J_carb_func =  
# J_jat_func = 
# J_prot_func = 

buffer = np.zeros(shape=(50,))

def F_vec(y_vec: np.array,t: float, param_vec: np.array):
    # свободные функции 
    J_carb = J_carb_func(t)
    J_prot = J_prot_func(t)
    J_fat  = J_fat_func(t)
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

    J_0 = j_0 * TG_pl
    J_1 = j_1 * Glu_ef
    J_2 = j_2 * KB_ef
    J_3 = j_3 * FFA_ef
    J_4 = j_4 * AA_ef

    # два - выходы мочевины и холестерола.

      #   Экскреция мочевины = UUN (Urine urea nitrogen) - мочевина в суточной моче, в среднем 570 ммоль/сут
      #   'EXCR_{Urea}': 570 (10**(-3))

      #    Холестерин: синтезируется 0,8 г/сут, приходит с пищей 0,4 г/сут, экскреция 1,2 г/сут

      #   'J_{cholesterol}': 0,4 г/сут
      #   'EXCR_{Cholesterol}': 1,2 г/сут

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

    # непостредственно вычисление вектора F(t) в точке t

    #                                 Метаболиты
    # 1. Adipocyte
    right_TG_a=A_7 - A_3
    right_AA_a=A_1 - A_17 - A_18 - A_19
    right_G6_a=A_4 - A_5 - A_6
    right_G3_a=2*A_5 + A_6 + A_9 - A_7 - A_8
    right_Pyr_a=A_8 + A_12 + A_19 - A_10 - A_11
    right_Ac_CoA_a=A_10 + A_14 + A_18 - 8*A_13 - A_16
    right_FA_CoA_a=A_2 + A_13 - 3*A_7
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
     #   '[O2]_{m}': r'
     #   'ANAEROB': r'
     #   'AEROB': r'
    # 4. Extracellular fluid

    # Diet-induced concentrations (нутриенты в крови):
    right_Glu_ef = J_carb + H_2 - H_3 - M_1 - A_4 - J_1
    right_AA_ef =  J_prot + M_6 - A_1 - H_1 - J_4 - M_5
    right_TG_pl =  J_fat + H_9 - J_0

    # Metabolome (метаболиты в крови):
    right_Glycerol_ef=    J_0 + A_3 - H_4
    right_FFA_ef= 3*J_0 + 3*A_3 - A_2 - H_8 - M_4 - J_3
    right_Lac_m=  M_2 - H_5
    right_KB_ef=  H_6 - M_3 - J_2

    # Excreted substances (мочевина, холестерин):
    right_Urea_ef=    J_4 + A_17 + A_18 + A_19 + M_17 + M_18 + M_19 + H_27 + H_28 + H_29
    right_Cholesterol_pl= H_7

    # Гормоны:
    right_INS= alpha * J_carb +beta * J_fat + gamma * J_prot - CL_INS * INS
    right_GLN = lambda_ * (1.0/np.minimum(Glu_ef, 0.001)) - CL_GLN * GLN
    right_CAM = sigma * HeartRate - CL_CAM * CAM
    right_Muscle_m = M_20 - M_21



    # output buffer
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
    return buffer
