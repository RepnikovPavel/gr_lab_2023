import re

code = '''
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
'''

# Заменить буквы M_number на соответствующие выражения
code = re.sub(r'M_(\d+)', lambda match: f'M_{match.group(1)} = {eval(f"M_{match.group(1)}")}', code)

# Заменить буквы H_number на соответствующие выражения
code = re.sub(r'H_(\d+)', lambda match: f'H_{match.group(1)} = {eval(f"H_{match.group(1)}")}', code)

# Заменить буквы A_number на соответствующие выражения
code = re.sub(r'A_(\d+)', lambda match: f'A_{match.group(1)} = {eval(f"A_{match.group(1)}")}', code)

# Теперь переменные M_number, H_number и A_number заменены на соответствующие выражения в коде
print(code)