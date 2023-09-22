# переменные выделяются '$' с обеих сторон
# сокращения выделяются '@' с обеих сторон
# функции выделяются '#' с обеих сторон
# врехний индекс запрещен
# вложенный нижний индекс запрещен
import os

import numpy as np

import MyocyteSystem.config as config

system_functions = {
    r'[KB-plasma]': {'py_name': 'KB_plasma',    'module': 'AdipocyteSystem.adipocyte_de_config'},
    r'F_{carb}':    {'py_name': 'F_carb',       'module': 'AdipocyteSystem.adipocyte_de_config'},
    r'F_{fat}':     {'py_name': 'F_fat',        'module': 'AdipocyteSystem.adipocyte_de_config'},
    r'F_{prot}':    {'py_name': 'F_prot',       'module': 'AdipocyteSystem.adipocyte_de_config'},
    r'[SP]_{myo}':  {'py_name': 'SP_myo',       'module': 'AdipocyteSystem.adipocyte_de_config'},
    r'f':           {'py_name': 'sigmoid',      'module': 'AdipocyteSystem.adipocyte_de_config'}
}

write_generated_code_to = os.path.join(config.project_path, 'AdipocyteSystem')
write_latex_view_of_system_to = os.path.join(config.project_path, 'AdipocyteSystem')
name_of_generated_pyhton_file = 'adipocyte_gen_des.py'


def KB_plasma(x):
    return 0.0

def F_carb(x):
    return np.exp(-1.0*x)*(10**(-3))

def F_fat(x):
    return np.exp(-2.0*x)*(10**(-3))

def F_prot(x):
    return np.exp(-3.0*x)*(10**(-3))

def SP_myo(x):
    #print()
    return 0.0

def sigmoid(x, alpha, beta, gamma, delta):
    # limitations
    # beta > alpha
    # delta >= gamma
    if x < alpha:
        return gamma
    if x > beta:
        return delta
    if alpha <= x <= beta:
        return (delta-gamma)/(beta-alpha)*x+(gamma*beta-delta*alpha)/(beta-alpha)


# параметры alpha beta gamma delta должны быть того же порядка как и аргументы для функции f
params_values = {
    r'$a_{0}$':             1,
    r'$a_{3}$':             1,
    r'$a_{5}$':             1,
    r'$a_{12}$':            1,
    r'$a_{13}$':            1,
    r'$a_{14}$':            1,
    r'$a_{15}$':            1,
    r'$a_{16}$':            1,
    r'$a_{17}$':            1,
    r'$a_{18}$':            1,
    r'$a_{21}$':            1,
    r'$a_{22}$':            1,
    r'$a_{23}$':            1,
    r'$a_{24}$':            1,
    r'$a_{25}$':            1,
    r'$a_{27}$':            1,
    r'$a_{sp}$':            1,
    r'$\alpha$':           1,
    r'$\beta$':            2,
    r'$\gamma$':           4,
    r'$\delta$':           3,
    r'$\alpha_{1}$':          1,
    r'$\alpha_{2}$':          1,
    r'$\alpha_{4}$':          1,
    r'$\alpha_{7}$':          1,
    r'$\alpha_{8}$':          1,
    r'$\alpha_{9}$':          1,
    r'$\alpha_{10}$':         1,
    r'$\alpha_{11}$':         1,
    r'$\alpha_{191}$':       1,
    r'$\alpha_{192}$':       1,
    r'$\alpha_{20}$':         1,
    r'$\alpha_{26}$':         1,
    r'$\alpha_{28}$':         1,
    r'$\beta_{1}$':          2,
    r'$\beta_{2}$':          2,
    r'$\beta_{4}$':          2,
    r'$\beta_{7}$':          2,
    r'$\beta_{8}$':          2,
    r'$\beta_{9}$':          2,
    r'$\beta_{10}$':         2,
    r'$\beta_{11}$':         2,
    r'$\beta_{191}$':       2,
    r'$\beta_{192}$':       2,
    r'$\beta_{20}$':         2,
    r'$\beta_{26}$':         2,
    r'$\beta_{28}$':         2,
    r'$\gamma_{1}$':          3,
    r'$\gamma_{2}$':          3,
    r'$\gamma_{4}$':          3,
    r'$\gamma_{7}$':          3,
    r'$\gamma_{8}$':          3,
    r'$\gamma_{9}$':          3,
    r'$\gamma_{10}$':         3,
    r'$\gamma_{11}$':         3,
    r'$\gamma_{191}$':       3,
    r'$\gamma_{192}$':       3,
    r'$\gamma_{20}$':         3,
    r'$\gamma_{26}$':         3,
    r'$\gamma_{28}$':         3,
    r'$\delta_{1}$':          4,
    r'$\delta_{2}$':          4,
    r'$\delta_{4}$':          4,
    r'$\delta_{7}$':          4,
    r'$\delta_{8}$':          4,
    r'$\delta_{9}$':          4,
    r'$\delta_{10}$':         4,
    r'$\delta_{11}$':         4,
    r'$\delta_{191}$':       4,
    r'$\delta_{192}$':       4,
    r'$\delta_{20}$':         4,
    r'$\delta_{26}$':         4,
    r'$\delta_{28}$':         4,
    r'$FFA_plasma_local$':    1,
    r'$LD$':                  1,
    r'$FFA_global$':          1,
    r'$J_carb$':              1,
    r'$J_fat$':               1,
    r'$J_prot$':              1,
    r'$CL$':                  1
}

start_point = {
    'D_{carb}':                 1.0,
    'D_{fat}':                  1.0,
    'D_{prot}':                 1.0,
    'J_{carb}':                 1.0,
    'J_{fat}':                  1.0, # mmol/L
    'J_{prot}':                 1.0,
    '[AA-food]':                1.0,
    '[AA]_{myo}':               1.0,# мкмоль/л
    '[ATP-c]_{myo}':            1.0, # 2 mmol/L
    '[ATP-m]_{myo}':            1.0, # 2 mmol/L
    '[Ac-CoA]_{myo}':           1.0, # ng/uL
    '[CO_2]_{myo}':             1.0, # 23-29 mmol/L
    '[Cytr]_{myo}':             1.0, # 75nm # https://doi.org/10.1074/jbc.274.23.16010
    '[FA-CoA]_{myo}':           1.0,
    '[FFA]_{myo}':              1.0,
    '[FFA^{loc}_{pl}]_{myo}':   1.0,
    '[G3P]_{myo}':              1.0,
    '[G6P]_{myo}':              1.0,
    '[GG]_{myo}':               1.0, #  8-61 МЕ/л # GGT- 33-48 IU/L 0.5 мкимоль/(с*л)
    '[Gluc-plasma]':            1.0, # mmol/L # КЛИНИЧЕСКОЕ ЗНАЧЕНИЕ ПОКАЗАТЕЛЕЙ УГЛЕВОДНОГО ОБМЕНА.doc
    '[Gluc]_{myo}':             1.0,  # 3.9(1.03) mmol/L #  DOI: 10.1016/j.ejogrb.2005.08.028 # 3,33—5,55 ммоль/л в крови
    '[Gly3P]_{myo}':            1.0,
    '[INS]':                    1.0, # 138 pmol/L
    '[LD]_{myo}':               1.0,
    '[Lac]_{myo}':              1.0,
    '[OAA]_{myo}':              1.0, # мкмоль/?
    '[Pyr]_{myo}':              1.0,
    '[TAG-plasma]':             1.0,
    '[TAG]_{myo}':              1.0,
    '[[H]f-m]_{myo}':           1.0, # 8 млмоль/?
    '[[H]n-c]_{myo}':           1.0, # 8 млмоль/?
    '[[H]nf-m]_{myo}':          1.0, # 8 млмоль/?
    '[muscle]_{myo}':           1.0
}

aliases_ = {
    '@A_0@': r'$a_{0}$',
    '@A_1@': r'#f([INS],$\alpha_{1}$,$\beta_{1}$,$\gamma_{1}$,$\delta_{1}$)# * [FFA^{loc}_{pl}]_{local}',
    '@A_2@': r'$m_{2}$ * [FFA]_{myo}',
    '@A_3@': r'$m_{3}$ * [FA-CoA]_{myo}',
    '@A_4@': r'$m_{4}$ * [Ac-CoA]_{myo}',
    '@A_5@': r'$m_{5}$ * [FA-CoA]_{myo} * [Gly3P]_{myo}',
    '@A_6@': r'$m_{6}$ * [TAG]_{myo}',
    '@A_7@': r'#f([INS],$\alpha_{7}$,$\beta_{7}$,$\gamma_{7}$,$\delta_{7}$)# * [Gluc]_{myo}',
    '@A_8@': r'#f([INS],$\alpha_{8}$,$\beta_{8}$,$\gamma_{8}$,$\delta_{8}$)# * [G6P]_{myo}',
    '@A_9@': r'#f([INS],$\alpha_{9}$,$\beta_{9}$,$\gamma_{9}$,$\delta_{9}$)# * [G6P]_{myo}',
    '@A_10@': r'$m_{10}$ * [GG]_{myo}',
    '@A_11@': r'$m_{11}$ * [G3P]_{myo} * [[H]n-c]_{myo} * [ATP-c]_{myo}',
    '@A_12@': r'#f([INS],$\alpha_{12}$,$\beta_{12}$,$\gamma_{12}$,$\delta_{12}$)#*#f([Ac-CoA]_{myo},$\alpha_{122}$,$\beta_{122}$,$\gamma_{122}$, $\delta_{122}$)#*[Pyr]_{myo}',
    '@A_13@': r'$m_{13}$ * #[KB-plasma](t)#',
    '@A_14@': r'$m_{14}$ * [AA]_{myo}',
    '@A_15@': r'$m_{15}$ * [AA]_{myo}',
    '@A_16@': r'$m_{16}$ * [AA]_{myo}',
    '@A_17@': r'#f([AA-food],$\alpha_{17}$,$\beta_{17}$,$\gamma_{17}$,$\delta_{17}$)# * [muscle]_{myo}',
    '@A_18@': r'$m_{18}$ * [AA]_{myo}',
    '@A_19@': r'#f([AA-food],$\alpha_{19}$,$\beta_{19}$,$\gamma_{19}$,$\delta_{19}$)# * [AA]_{myo}',
    '@A_20@': r'$m_{20}$ * [AA-food]',
    '@A_21@': r'$m_{21}$ * [Pyr]_{myo} * [[H]n-c]_{myo}',
    '@A_22@': r'$m_{22}$ * [Lac]_{myo}',
    '@A_23@': r'#f([INS],$\alpha_{23}$,$\beta_{23}$,$\gamma_{23}$,$\delta_{23}$)#* [Gluc-plasma]',
    '@A_24@': r'$m_{24}$ * [Cytr]_{myo}',
    '@A_25@': r'$m_{25}$ * [[H]nf-m]_{myo}',
    '@A_26@': r'$m_{26}$ * [ATP-m]_{myo}',
    '@A_27@': r'$m_{27}$ * [G3P]_{myo} * [[H]n-c]_{myo}',
    '@A_28@': r''
}
des_str_ = {
    '[FFA_{pl}^{loc}]_{adp}':   '@A_0@ - @A_1@ - SP_a',
    '[FA-CoA]_{adp}':           '@A_1@ + @A_6@ - @A_2@ - A_carn',
    '[Gly3P]_{adp}':            '@A_27@ - @A_2@',
    '[TAG]_{adp}':              '@A_2@ - @A_3@',
    '[LD]_{adp}':               '@A_3@ - @A_4@',
    '[FFA]_{adp}':              '3*@A_4@ - @A_5@ - @A_6@ + @A_25@',
    '[G3P]_{adp}':              '2*@A_9@ + @A_12@ + @A_26@ - @A_13@ - @A_27@',
    '[G6P]_{adp}':              '@A_8@ + @A_11@ - @A_9@ - @A_10@ - @A_12@',
    '[Gluc]_{adp}':             '@A_7@ - @A_8@',
    '[GG]_{adp}':               '@A_10@ - @A_11@',
    '[Pyr]_{adp}':              '@A_14@ + @A_13@ + @A_24@ - @A_19@ - @A_26@ - @A_17@',
    '[Lac]_{adp}':              '@A_17@ - @A_18@',
    '[[H]n-c]_{adp}':           '@A_13@ - @A_17@',
    '[AA]_{adp}':               '@A_28@ - @A_14@ - @A_15@ - @A_16@',
    '[Ac-CoA]_{adp}':           '@A_19@ + @A_15@ - @A_20@ - @A_21@ - @A_25@',
    '[Mal-CoA]_{adp}':          '@A_20@ - @A_25@',
    '[Cytr]_{adp}':             '@A_21@ - @A_22@',
    '[OAA]_{adp}':              '@A_16@ + @A_22@ - @A_21@ - @A_24@',
    '[CO_{2}]_{adp}':           '@A_22@ + @A_19@',
    '[[H]nf-m]_{adp}':          '@A_22@ - @A_23@',
    '[ATP]_{adp}':              '@A_23@ + @A_13@',
    '[NADPH]_{adp}':            '@A_12@ + @A_24@ - @A_25@'
}
