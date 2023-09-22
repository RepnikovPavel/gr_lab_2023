import numpy as np
import numba 
from sysv1_de_config import F_carb
from sysv1_de_config import F_fat
from sysv1_de_config import F_prot
from sysv1_de_config import sigmoid


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

def F_vec(y_vec: np.array,t: float, param_vec: np.array):
    # Y_{t} values
    Glu_ef  = y_vec[0]                  
    AA_ef   = y_vec[1]                   
    Glycerol_ef     = y_vec[2]             
    FFA_ef  = y_vec[3]                 
    Lac_m   = y_vec[4]                   
    KB_ef   = y_vec[5]                  
    Cholestrol_pl   = y_vec[6]           
    TG_pl   = y_vec[7]                   
    G6_a    = y_vec[8]                    
    G3_a    = y_vec[9]            
    Pyr_a   = y_vec[10]           
    Ac_CoA_a    = y_vec[11]        
    FA_CoA_a    = y_vec[12]        
    Cit_a   = y_vec[13]           
    OAA_a   = y_vec[14]           
    AA_a    = y_vec[15]            
    NADPH_a     = y_vec[16]         
    TG_a    = y_vec[17]                     
    GG_m    = y_vec[18]                     
    G6_m    = y_vec[19]            
    G3_m    = y_vec[20]            
    Pyr_m   = y_vec[21]           
    Ac_CoA_m    = y_vec[22]        
    FA_CoA_m    = y_vec[23]        
    Cit_m   = y_vec[24]           
    OAA_m   = y_vec[25]           
    H_cyt_m     = y_vec[26]         
    H_mit_m     = y_vec[27]         
    AA_m    = y_vec[28]            
    Muscle_m    = y_vec[29]                 
    CO2_m   = y_vec[30]           
    H20_m   = y_vec[31]           
    ATP_cyt_m   = y_vec[32]        
    ATP_mit_m   = y_vec[33]        
    GG_h    = y_vec[34]                    
    G6_h    = y_vec[35]            
    G3_h    = y_vec[36]            
    Pyr_h   = y_vec[37]           
    Ac_CoA_h    = y_vec[38]        
    FA_CoA_h    = y_vec[39]        
    MVA_h   = y_vec[40]           
    Cit_h   = y_vec[41]           
    OAA_h   = y_vec[42]           
    NADPH_h     = y_vec[43]         
    AA_h    = y_vec[44]            
    TG_h    = y_vec[45]                    
    Urea_ef     = y_vec[46]                 



    '@f([INS]_{j0})@': r'1', #
    '@f([INS]_{a2})@': r'1', # r'#f([INS],$\alpha_{a2}$,$\beta_{a2}$,$\gamma_{a2}$,$\delta_{a2}$)#',
    '@f([INS]_{a4})@': r'1', # r'#f([INS],$\alpha_{a4}$,$\beta_{a4}$,$\gamma_{a4}$,$\delta_{a4}$)#',
    '@f([INS]_{a5})@': r'1', # r'#f([INS],$\alpha_{a5}$,$\beta_{a5}$,$\gamma_{a5}$,$\delta_{a5}$)#',
    '@f([INS]_{a7})@': r'1', # r'#f([INS],$\alpha_{a7}$,$\beta_{a7}$,$\gamma_{a7}$,$\delta_{a7}$)#',
    '@f([INS]_{a10})@': r'1', # r'#f([INS],$\alpha_{a10}$,$\beta_{a10}$,$\gamma_{a10}$,$\delta_{a10}$)#',
    '@f([INS]_{a12})@': r'1', #
    '@f([INS]_{a13})@': r'1', #
    '@f([INS]_{a14})@': r'1', #
    '@f([INS]_{m1})@': r'1', # r'#f([INS],$\alpha_{m1}$,$\beta_{m1}$,$\gamma_{m1}$,$\delta_{m1}$)#',
    '@f([INS]_{m7})@': r'1', # r'#f([INS],$\alpha_{m7}$,$\beta_{m7}$,$\gamma_{m7}$,$\delta_{m7}$)#',
    '@f([INS]_{m9})@': r'1', # r'#f([INS],$\alpha_{m9}$,$\beta_{m9}$,$\gamma_{m9}$,$\delta_{m9}$)#',
    '@f([INS]_{m11})@': r'1', # r'#f([INS],$\alpha_{m11}$,$\beta_{m11}$,$\gamma_{m11}$,$\delta_{m11}$)#',
    '@f([INS]_{h3})@': r'1', # r'#f([INS],$\alpha_{h3}$,$\beta_{h3}$,$\gamma_{h3}$,$\delta_{h3}$)#',
    '@f([INS]_{h6})@': r'1', # чем меньше АУК инсулина за последние 3 часа, тем выше скорость H6
    '@f([INS]_{h7})@': r'1', # r'#f([INS],$\alpha_{h7}$,$\beta_{h7}$,$\gamma_{h7}$,$\delta_{h8}$)#',
    '@f([INS]_{h10})@': r'1', # r'#f([INS],$\alpha_{h10}$,$\beta_{h10}$,$\gamma_{h10}$,$\delta_{h10}$)#',
    '@f([INS]_{h12})@': r'1', # r'#f([INS],$\alpha_{h12}$,$\beta_{h12}$,$\gamma_{h12}$,$\delta_{h12}$)#',
    '@f([INS]_{h16})@': r'1', # r'#f([INS],$\alpha_{h167}$,$\beta_{h16}$,$\gamma_{h16}$,$\delta_{h16}$)#',
    '@f([INS]_{h19})@': r'1', #
    '@f([INS]_{h24})@': r'1', # r'#f([INS],$\alpha_{h24}$,$\beta_{h24}$,$\gamma_{h24}$,$\delta_{h24}$)#',
    '@f([INS]_{h26})@': r'1', # r'#f([INS],$\alpha_{h26}$,$\beta_{h26}$,$\gamma_{h26}$,$\delta_{h26}$)#',
    '@f([GLN]_{a3})@': r'1', #r'#f([G6]_{ef},$\lambda_{a3}$)#',
    '@f([GLN]_{a9})@': r'1', #r'#f([G6]_{ef},$\lambda_{a9}$)#',
    '@f([GLN]_{a11})@': r'1', #r'#f([G6]_{ef},$\lambda_{a11}$)#',
    '@f([GLN]_{m8})@': r'1', #r'#f([G6]_{ef},$\lambda_{m8}$)#',
    '@f([GLN]_{h2})@': r'1', #r'#f([G6]_{ef},$\lambda_{h2}$)#',
    '@f([GLN]_{h11})@': r'1', #r'#f([G6]_{ef},$\lambda_{h11}$)#',
    '@f([GLN]_{h13})@': r'1', #r'#f([G6]_{ef},$\lambda_{h13}$)#',
    '@f([GLN]_{h23})@': r'1',  # r'#f([G6]_{ef},$\lambda_{h23}$)#',
    '@f([GLN]_{h25})@': r'1', #r'#f([G6]_{ef},$\lambda_{h25}$)#',
    '@f([CAM]_{a3})@': r'1', #r'#f([Heart Rate],$\sigma_{a3}$)#',
    '@f([CAM]_{a11})@': r'1',  # r'#f([Heart Rate],$\sigma_{a11}$)#',
    '@f([CAM]_{m8})@': r'1', #r'#f([Heart Rate],$\sigma_{m8}$)#',
    '@f([CAM]_{h11})@': r'1', #r'#f([Heart Rate],$\sigma_{h11}$)#',
    '@f([CAM]_{h25})@': r'1', #r'#f([Heart Rate],$\sigma_{h25}$)#',
    '@J_0@': r'$j_{0}$ * [TG]_{pl} * #f([INS]_{j0})#',
    '@J_1@': r'$j_{1}$ * [Glu]_{ef}',
    '@J_2@': r'$j_{2}$ * [KB]_{ef}',
    '@J_3@': r'$j_{3}$ * [FFA]_{ef}',
    '@J_4@': r'$j_{4}$ * [AA]_{ef}',
    # три - входы БЖУ;

      #  '@J_{carb}@':
      #  '@J_{prot}@':
      #  '@J_{fat}@':

    # Ниже в коде есть такая часть:

        # Вход нутриентов (часть старого кода)

        #    'D_{carb}': r'($a_{carb}$ * #F_{carb}(t)# - D_{carb}) / $\tau_{carb}$',
        #    'J_{carb}': r'(D_{carb} - J_{carb}) / $\tau_{carb}$',
        #    'D_{fat}': r'($a_{fat}$ * #F_{fat}(t)# - D_{fat}) / $\tau_{fat}$',
        #    'J_{fat}': r'(D_{fat} - J_{fat}) / $\tau_{fat}$',
        #    'D_{prot}': r'($a_{prot}$ * #F_{prot}(t)# - D_{prot}) / $\tau_{prot}$',
        #    'J_{prot}': r'(D_{prot} - J_{prot}) / $\tau_{prot}$',


    # два - выходы мочевины и холестерола.

      #   Экскреция мочевины = UUN (Urine urea nitrogen) - мочевина в суточной моче, в среднем 570 ммоль/сут
      #   '@EXCR_{Urea}@': 570 (10**(-3))

      #    Холестерин: синтезируется 0,8 г/сут, приходит с пищей 0,4 г/сут, экскреция 1,2 г/сут

      #   '@J_{cholesterol}@': 0,4 г/сут
      #   '@EXCR_{Cholesterol}@': 1,2 г/сут


    # 2. Myocyte

    '@M_1@': r'$m_{1}$ * [Glu]_{ef} * #f([INS]_{m1})#',
    '@M_2@': r'$m_{2}$ * [Pyr]_{m}',
    '@M_3@': r'$m_{3}$ * [KB]_{ef}',
    '@M_4@': r'$m_{4}$ * [FFA]_{ef} * #f([INS]_{m4})#',
    '@M_5@': r'$m_{5}$ * [AA]_{ef}',
    '@M_6@': r'$m_{6}$ * [AA]_{m}',
    '@M_7@': r'$m_{7}$ * [G6]_{m} * #f([INS]_{m7})#',
    '@M_8@': r'$m_{8}$ * [GG]_{m} * #f([GLN]_{m8})# * #f([CAM]_{m8})#',
    '@M_9@': r'$m_{9}$ * [G6]_{m} * #f([INS]_{m9})#',
    '@M_10@': r'$m_{10}$ * [G3]_{m}',
    '@M_11@': r'$m_{11}$ * [Pyr]_{m} * #f([INS]_{m11})#',
    '@M_12@': r'$m_{12}$ * [FA-CoA]_{m}',
    '@M_13@': r'$m_{13}$ * [Ac-CoA]_{m} * [OAA]_{m}',
    '@M_14@': r'$m_{14}$ * [Cit]_{m}',
    '@M_15@': r'$m_{15}$ * [H-cyt]_{m}',
    '@M_16@': r'$m_{16}$ * [H-mit]_{m}', # * [O2]',
    '@M_17@': r'$m_{17}$ * [AA]_{m}',
    '@M_18@': r'$m_{18}$ * [AA]_{m}',
    '@M_19@': r'$m_{19}$ * [AA]_{m}',
    '@M_20@': r'$m_{20}$ * [AA]_{m}',
    '@M_21@': r'$m_{21}$ * [Muscle]_{m}',



    #3. Adipocyte

    '@A_1@': r'$a_{1}$ * [AA]_{ef}',
    '@A_2@': r'$a_{2}$ * [FFA]_{ef} * #f([INS]_{a2})#',
    '@A_3@': r'$a_{3}$ * [TG]_{a} * #f([GLN]_{a3})# * #f([CAM]_{a3})#',
    '@A_4@': r'$a_{4}$ * [Glu]_{ef} * #f([INS]_{a4})#',
    '@A_5@': r'$a_{5}$ * [G6]_{a} * #f([INS]_{a5})#',
    '@A_6@': r'$a_{6}$ * [G6]_{a}',
    '@A_7@': r'$a_{7}$ * [G3]_{a} * [FA-CoA]_{a} * #f([INS]_{a7})#',
    '@A_8@': r'$a_{8}$ * [G3]_{a}',
    '@A_9@': r'$a_{9}$ * [OAA]_{a} * #f([GLN]_{a9})#',
    '@A_10@': r'$a_{10}$ * [Pyr]_{a} * #f([INS]_{a10})#',
    '@A_11@': r'$a_{11}$ * [Pyr]_{a} * #f([GLN]_{a11})#',
    '@A_12@': r'$a_{12}$ * [OAA]_{a}',
    '@A_13@': r'$a_{13}$ * [Ac-CoA]_{a} * [NADPH]_{a}',
    '@A_14@': r'$a_{14}$ * [Cit]_{a} * #f([INS]_{a14})#',
    '@A_15@': r'$a_{15}$ * [Cit]_{a}',
    '@A_16@': r'$a_{16}$ * [OAA]_{a} * [Cit]_{a}',
    '@A_17@': r'$a_{17}$ * [AA]_{a}',
    '@A_18@': r'$a_{18}$ * [AA]_{a}',
    '@A_19@': r'$a_{19}$ * [AA]_{a}',

    #4. Hepatocyte

    '@H_1@': r'$h_{1}$ * [AA]_{ef}',
    '@H_2@': r'$h_{2}$ * [G6]_{h} * #f([GLN]_{h2})#',
    '@H_3@': r'$h_{3}$ * [Glu]_{ef} * #f([INS]_{h3})#',
    '@H_4@': r'$h_{4}$ * [Glycerol]_{ef}',
    '@H_5@': r'$h_{5}$ * [Lac]_{m}',
    '@H_6@': r'$h_{6}$ * [Ac-CoA]_{h} * #f([INS]_{h6})#',
    '@H_7@': r'$h_{7}$ * [MVA]_{h} * #f([INS]_{h7})#',
    '@H_8@': r'$h_{8}$ * [FFA]_{ef}',
    '@H_9@': r'$h_{9}$ * [TG]_{h}',
    '@H_10@': r'$h_{10}$ * [G6]_{h} * #f([INS]_{h10})#',
    '@H_11@': r'$h_{11}$ * [GG]_{h} * #f([GLN]_{h11})# * #f([CAM]_{h11})#',
    '@H_12@': r'$h_{12}$ * [G6]_{h} * #f([INS]_{h12})#',
    '@H_13@': r'$h_{13}$ * [G3]_{h}',
    '@H_14@': r'$h_{14}$ * [G6]_{h}',
    '@H_15@': r'$h_{15}$ * [G3]_{h}',
    '@H_16@': r'$h_{16}$ * [Pyr]_{h} * #f([INS]_{h16})#',
    '@H_17@': r'$h_{17}$ * [Ac-CoA]_{h} * [NADPH]_{h} * #f([INS]_{h17})#',
    '@H_18@': r'$h_{18}$ * [FA-CoA]_{h}  * #f([GLN]_{h18})#',
    '@H_19@': r'$h_{19}$ * [Ac-CoA]_{h} * [NADPH]_{h} * #f([INS]_{h19})#',
    '@H_20@': r'$h_{20}$ * [G3]_{h} * [FA-CoA]_{h} * #f([INS]_{h20})#',
    '@H_21@': r'$h_{21}$ * [Ac-CoA]_{h} * [OAA]_{h}',
    '@H_22@': r'$h_{22}$ * [Cit]_{h}',
    '@H_23@': r'$h_{23}$ * [OAA]_{h} * #f([GLN]_{h23})#',
    '@H_24@': r'$h_{24}$ * [OAA]_{h} * #f([INS]_{h24})#',
    '@H_25@': r'$h_{25}$ * [Pyr]_{h} * #f([GLN]_{h25})# * #f([CAM]_{h25})#',
    '@H_26@': r'$h_{26}$ * [AA]_{h}',
    '@H_27@': r'$h_{27}$ * [AA]_{h}',
    '@H_28@': r'$h_{28}$ * [AA]_{h}',
    '@H_29@': r'$h_{29}$ * [AA]_{h}',


}
des_str_ = {

    # Метаболиты

    # 1. Adipocyte

    r'[TG]_{a}': r'@A_7@ - @A_3@',
    r'[AA]_{a}': r'@A_1@ - @A_17@ - @A_18@ - @A_19@',
    r'[G6]_{a}': r'@A_4@ - @A_5@ - @A_6@',
    r'[G3]_{a}': r'2*@A_5@ + @A_6@ + @A_9@ - @A_7@ - @A_8@',
    r'[Pyr]_{a}': r'@A_8@ + @A_12@ + @A_19@ - @A_10@ - @A_11@',
    r'[Ac-CoA]_{a}': r'@A_10@ + @A_14@ + @A_18@ - 8*@A_13@ - @A_16@',
    r'[FA-CoA]_{a}': r'@A_2@ + @A_13@ - 3*@A_7@',
    r'[Cit]_{a}': r'@A_16@ - @A_14@ - @A_15@',
    r'[OAA]_{a}': r'@A_11@ + @A_14@ + @A_15@ + @A_17@ - @A_9@ - @A_12@ - @A_16@ ',
    r'[NADPH]_{a}': r'@A_6@ + @A_12@ - 14*@A_13@',

    # 2. Hepatocyte

    r'[GG]_{h}': r'@H_10@ - @H_11@',
    r'[G6]_{h}': r'@H_3@ + @H_11@ + @H_13@ - @H_2@ - @H_10@ - @H_12@ - @H_14@',
    r'[G3]_{h}': r'@H_4@ + 2*@H_12@ + @H_14@ + @H_23@ - 2*@H_13@ - @H_15@ - @H_20@',
    r'[Pyr]_{h}': r'@H_5@ + @H_15@ + @H_24@ + @H_29@ - @H_16@ - @H_25@',
    r'[Ac-CoA]_{h}': r'@H_16@ + 8*@H_18@ + @H_26@ + @H_27@ - 3*@H_17@ - 8*@H_19@',
    r'[FA-CoA]_{h}': r'@H_8@ + @H_19@ - @H_18@ - 3*@H_20@',
    r'[TG]_{h}': r'@H_20@ - @H_9@',
    r'[MVA]_{h}': r'@H_17@ - @H_7@',
    r'[OAA]_{h}': r'@H_22@ + @H_25@ + @H_26@  + @H_28@ - @H_21@ - @H_23@ - @H_24@',
    r'[Cit]_{h}': r'@H_21@ - @H_22@ - @H_26@',
    r'[AA]_{h}': r'@H_1@ - @H_27@ - @H_28@ - @H_29@',
    r'[NADPH]_{h}': r'6*@H_14@ + @H_24@ - 14*@H_19@',

    # 3. Myocyte

    r'[GG]_{m}': r'@M_7@ - @M_8@',
    r'[G6]_{m}': r'@M_1@ + @M_8@ - @M_7@ - @M_9@',
    r'[G3]_{m}': r'2*@M_9@ - @M_10@',
    r'[Pyr]_{m}': r'@M_10@ + @M_17@ - @M_11@ - @M_2@',
    r'[Ac-CoA]_{m}': r'2*@M_3@ + @M_11@ + 8*@M_12@ + @M_18@ - @M_13@',
    r'[FA-CoA]_{m}': r'@M_4@ - @M_12@',
    r'[AA]_{m}': r'@M_5@ + @M_21@ - @M_6@ - @M_17@ - @M_18@ - @M_19@ - @M_20@',
    r'[Cit]_{m}': r'@M_13@ - @M_14@',
    r'[OAA]_{m}': r'@M_14@ + @M_19@ - @M_13@',
    r'[H-cyt]_{m}': r'2*@M_10@ - @M_2@ - @M_15@',
    r'[H-mit]_{m}': r'@M_3@ + 14*@M_12@ + @M_15@ - @M_16@',
    r'[CO2]_{m}': r'@M_11@ + 2*@M_14@',
    r'[H2O]_{m}': r'@M_16@',

    r'[ATP-cyt]_{m}': r'@M_10@', # Anaerob
    r'[ATP-mit]_{m}': r'2*@M_16@', # Aerob


     # Далее три величины, которые требуют обсуждения.
     # Работа (аэробная и анаэробная) - является ли она интегральной величиной?
     # Или это просто "депо" ккал, которое должно обнуляться каждую полночь?
     # Зависит ли концентрация кислорода от ЧСС, или он просто всегда в избытке (равен единице)?

     #   '[O2]_{m}': r'',
     #   'ANAEROB': r'',
     #   'AEROB': r'',


    # 4. Extracellular fluid

    # Плазма крови для TG и межклеточная жидкость для всех остальных.
    # Отличие - в объеме: плазма - 5,5 л, вся МЖ - около 10 л).

    # Diet-induced concentrations (нутриенты в крови):
    r'[Glu]_{ef}': r'J_{carb} + @H_2@ - @H_3@ - @M_1@ - @A_4@ - @J_1@',
    r'[AA]_{ef}': r'J_{prot} + @M_6@ - @A_1@ - @H_1@ - @J_4@ - @M_5@',
    r'[TG]_{pl}': r'J_{fat} + @H_9@ - @J_0@',

    # Metabolome (метаболиты в крови):
    r'[Glycerol]_{ef}': r'@J_0@ + @A_3@ - @H_4@',
    r'[FFA]_{ef}': r'3*@J_0@ + 3*@A_3@ - @A_2@ - @H_8@ - @M_4@ - @J_3@',
    r'[Lac]_{m}': r'@M_2@ - @H_5@',
    r'[KB]_{ef}': r'@H_6@ - @M_3@ - @J_2@',
    r'[Cholesterol]_{pl}': r'@H_7@',
    r'[Urea]_{ef}': r'@J_4@ + @A_17@ + @A_18@ + @A_19@ + @M_17@ + @M_18@ + @M_19@ + @H_27@ + @H_28@ + @H_29@',

    # Excreted substances (мочевина, холестерин):
    # Для этих веществ нужно добавить в уравнение экскрецию как вычитаемую константу EXCR_{},
    # либо обнулять их депо каждую полночь.

    # '[Urea]_{ef}': r'@J_4@ + @A_17@ + @A_18@ + @A_19@ + @M_17@ + @M_18@ + @M_19@ + @H_27@ + @H_28@ + @H_29@' #- EXCR_{Urea},
    #  '[Cholesterol]_{pl}': r'@H_6@ + J_{cholesterol}@ - EXCR_{Cholesterol}',



    # Вход нутриентов (здесь ничего не меняю, это часть старого кода)
    r'D_{carb}': r'($a_{carb}$ * #F_{carb}(t)# - D_{carb}) / $\tau_{carb}$',
    r'J_{carb}': r'(D_{carb} - J_{carb}) / $\tau_{carb}$',
    r'D_{fat}': r'($a_{fat}$ * #F_{fat}(t)# - D_{fat}) / $\tau_{fat}$',
    r'J_{fat}': r'(D_{fat} - J_{fat}) / $\tau_{fat}$',
    r'D_{prot}': r'($a_{prot}$ * #F_{prot}(t)# - D_{prot}) / $\tau_{prot}$',
    r'J_{prot}': r'(D_{prot} - J_{prot}) / $\tau_{prot}$',

    # Гормоны:

    r'[INS]': r'$\alpha$ * J_{carb} +$\beta$ * J_{fat} + $\gamma$ * J_{prot} - $CL_{INS}$ * [INS]',

    # Добавим две новые функции гормонов (их надо обсудить и исправить):

    #'[GLN]': r'$\lambda$ * (1/[Glu]_{ef}) - $CL_{GLN}$ * [GLN]',
    #'[CAM]': r'$\sigma$ * [Heart Rate] - $CL_{CAM}$ * [CAM]'

    # Глюкагон [GLN] обратно зависим от концентрации [Glu]_{ef} (глюкозы в межклет жидкости).
    # GLN Выделяется при [Glu]_{ef} < 4 mmol/l.

    # Андреналин/норадреналин [CAM] можно отслеживать по пульсу.
    # Если пульс выше возрастной нормы, значит, [CAM] повышен.

    # Клиренс у всех трёх гормонов: полный - 5 минут, 50% - около 1 минуты. Падает экспоненциально.