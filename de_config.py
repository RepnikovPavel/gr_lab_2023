# переменные выделяются '$' с обеих сторон
# сокращения выделяются '@' с обеих сторон
# функции выделяются '#' с обеих сторон
# врехний индекс запрещен
# вложенный нижний индекс запрещен
import numpy as np


system_functions = {
    r'[KB-plasma]': {'py_name': 'KB_plasma',    'module': 'de_config'},
    r'F_{carb}':    {'py_name': 'F_carb',       'module': 'de_config'},
    r'F_{fat}':     {'py_name': 'F_fat',        'module': 'de_config'},
    r'F_{prot}':    {'py_name': 'F_prot',       'module': 'de_config'},
    r'[SP]_{myo}':  {'py_name': 'SP_myo',       'module': 'de_config'},
    r'f':           {'py_name': 'sigmoid',      'module': 'de_config'}
}


def KB_plasma(x):
    return 0.0

def F_carb(x):
    return np.exp(-1.0*x)*(10**(-3))

def F_fat(x):
    return np.exp(-2.0*x)*(10**(-3))

def F_prot(x):
    return np.exp(-3.0*x)*(10**(-3))

def SP_myo(x):
    return 0.0

def sigmoid(x, alpha, beta, gamma, delta):
    if x < alpha:
        return gamma
    if x > beta:
        return delta
    if alpha <= x <= beta:
        return (delta-gamma)/(beta-alpha)*x+(gamma*beta-delta*alpha)/(beta-alpha)


params_values = {
    '$CL$':               1.0,
    r'$\alpha$':          1.0,
    r'$\alpha_{1}$':      1.0,
    r'$\alpha_{12}$':     1.0,
    r'$\alpha_{122}$':    1.0,
    r'$\alpha_{17}$':     1.0,
    r'$\alpha_{19}$':     1.0,
    r'$\alpha_{23}$':     1.0,
    r'$\alpha_{31}$':     1.0,
    r'$\alpha_{7}$':      1.0,
    r'$\alpha_{8}$':      1.0,
    r'$\alpha_{9}$':      1.0,
    r'$\beta$':           1.0,
    r'$\beta_{1}$':       1.0,
    r'$\beta_{12}$':      1.0,
    r'$\beta_{122}$':     1.0,
    r'$\beta_{17}$':      1.0,
    r'$\beta_{19}$':      1.0,
    r'$\beta_{23}$':      1.0,
    r'$\beta_{31}$':      1.0,
    r'$\beta_{7}$':       1.0,
    r'$\beta_{8}$':       1.0,
    r'$\beta_{9}$':       1.0,
    r'$\delta_{1}$':      1.0,
    r'$\delta_{12}$':     1.0,
    r'$\delta_{122}$':    1.0,
    r'$\delta_{17}$':     1.0,
    r'$\delta_{19}$':     1.0,
    r'$\delta_{23}$':     1.0,
    r'$\delta_{31}$':     1.0,
    r'$\delta_{7}$':      1.0,
    r'$\delta_{8}$':      1.0,
    r'$\delta_{9}$':      1.0,
    r'$\gamma$':          1.0,
    r'$\gamma_{1}$':      1.0,
    r'$\gamma_{12}$':     1.0,
    r'$\gamma_{122}$':    1.0,
    r'$\gamma_{17}$':     1.0,
    r'$\gamma_{19}$':     1.0,
    r'$\gamma_{23}$':     1.0,
    r'$\gamma_{31}$':     1.0,
    r'$\gamma_{7}$':      1.0,
    r'$\gamma_{8}$':      1.0,
    r'$\gamma_{9}$':      1.0,
    r'$\tau_{carb}$':     1.0,
    r'$\tau_{fat}$':      1.0,
    r'$\tau_{prot}$':     1.0,
    '$a_{carb}$':         1.0,
    '$a_{fat}$':          1.0,
    '$a_{prot}$':         1.0,
    '$m_{0}$':            1.0,
    '$m_{10}$':           1.0,
    '$m_{11}$':           1.0,
    '$m_{13}$':           1.0,
    '$m_{14}$':           1.0,
    '$m_{15}$':           1.0,
    '$m_{16}$':           1.0,
    '$m_{18}$':           1.0,
    '$m_{2}$':            1.0,
    '$m_{20}$':           1.0,
    '$m_{21}$':           1.0,
    '$m_{22}$':           1.0,
    '$m_{24}$':           1.0,
    '$m_{25}$':           1.0,
    '$m_{26}$':           1.0,
    '$m_{27}$':           1.0,
    '$m_{28}$':           1.0,
    '$m_{29}$':           1.0,
    '$m_{3}$':            1.0,
    '$m_{30}$':           1.0,
    '$m_{4}$':            1.0,
    '$m_{5}$':            1.0,
    '$m_{6}$':            1.0
}

# моделирование инсулина #doi: 10.1016/S1674-8301(10)60048-6
# некоторые единицы https://unitslab.com/ru/node/40 с базой

start_point = {
    'D_{carb}':                 0.0,
    'D_{fat}':                  0.0,
    'D_{prot}':                 0.0,
    'J_{carb}':                 0.0,
    'J_{fat}':                  0.0, # mmol/L
    'J_{prot}':                 0.0,
    '[AA-food]':                0.0,
    '[AA]_{myo}':               100*(10**(-6)),# мкмоль/л
    '[ATP-c]_{myo}':            2*(10**(-3)), # 2 mmol/L
    '[ATP-m]_{myo}':            2*(10**(-3)), # 2 mmol/L
    '[Ac-CoA]_{myo}':           0.0, # ng/uL
    '[CO_2]_{myo}':             26*(10**(-3)), # 23-29 mmol/L
    '[Cytr]_{myo}':             0.0, # 75nm # https://doi.org/10.1074/jbc.274.23.16010
    '[FA-CoA]_{myo}':           0.0,
    '[FFA]_{myo}':              0.0,
    '[FFA^{loc}_{pl}]_{myo}':   0.0,
    '[G3P]_{myo}':              0.0,
    '[G6P]_{myo}':              0.0,
    '[GG]_{myo}':               0.5*(10**(-6)), #  8-61 МЕ/л # GGT- 33-48 IU/L 0.5 мкимоль/(с*л)
    '[Gluc-plasma]':            4.99*(10**(-3)), # mmol/L # КЛИНИЧЕСКОЕ ЗНАЧЕНИЕ ПОКАЗАТЕЛЕЙ УГЛЕВОДНОГО ОБМЕНА.doc
    '[Gluc]_{myo}':             3.9*(10**(-3)),  # 3.9(1.03) mmol/L #  DOI: 10.1016/j.ejogrb.2005.08.028 # 3,33—5,55 ммоль/л в крови
    '[Gly3P]_{myo}':            0.0,
    '[INS]':                    138*(10**(-12)), # 138 pmol/L
    '[LD]_{myo}':               0.0,
    '[Lac]_{myo}':              0.0,
    '[OAA]_{myo}':              100*(10**(-6)), # мкмоль/?
    '[Pyr]_{myo}':              0.0,
    '[TAG-plasma]':             0.0,
    '[TAG]_{myo}':              0.0,
    '[[H]f-m]_{myo}':           8*(10**(-3)), # 8 млмоль/?
    '[[H]n-c]_{myo}':           8*(10**(-3)), # 8 млмоль/?
    '[[H]nf-m]_{myo}':          8*(10**(-3)), # 8 млмоль/?
    '[muscle]_{myo}':           0.0
}



aliases_ = {
    '@M_0@': r'$m_{0}$ * [TAG-plasma]',
    '@M_1@': r'#f([INS],$\alpha_{1}$,$\beta_{1}$,$\gamma_{1}$,$\delta_{1}$)# * [FFA^{loc}_{pl}]_{myo}',
    '@M_2@': r'$m_{2}$ * [FFA]_{myo}',
    '@M_3@': r'$m_{3}$ * [FA-CoA]_{myo}',
    '@M_4@': r'$m_{4}$ * [Ac-CoA]_{myo}',
    '@M_5@': r'$m_{5}$ * [FA-CoA]_{myo} * [Gly3P]_{myo}',
    '@M_6@': r'$m_{6}$ * [TAG]_{myo}',
    '@M_7@': r'#f([INS],$\alpha_{7}$,$\beta_{7}$,$\gamma_{7}$,$\delta_{7}$)# * [Gluc]_{myo}',
    '@M_8@': r'#f([INS],$\alpha_{8}$,$\beta_{8}$,$\gamma_{8}$,$\delta_{8}$)# * [G6P]_{myo}',
    '@M_9@': r'#f([INS],$\alpha_{9}$,$\beta_{9}$,$\gamma_{9}$,$\delta_{9}$)# * [G6P]_{myo}',
    '@M_10@': r'$m_{10}$ * [GG]_{myo}',
    '@M_11@': r'$m_{11}$ * [G3P]_{myo} * [[H]n-c]_{myo} * [ATP-c]_{myo}',
    '@M_12@': r'#f([INS],$\alpha_{12}$,$\beta_{12}$,$\gamma_{12}$,$\delta_{12}$)#*#f([Ac-CoA]_{myo},$\alpha_{122}$,$\beta_{122}$,$\gamma_{122}$, $\delta_{122}$)#*[Pyr]_{myo}',
    '@M_13@': r'$m_{13}$ * #[KB-plasma](t)#',
    '@M_14@': r'$m_{14}$ * [AA]_{myo}',
    '@M_15@': r'$m_{15}$ * [AA]_{myo}',
    '@M_16@': r'$m_{16}$ * [AA]_{myo}',
    '@M_17@': r'#f([AA-food],$\alpha_{17}$,$\beta_{17}$,$\gamma_{17}$,$\delta_{17}$)# * [muscle]_{myo}',
    '@M_18@': r'$m_{18}$ * [AA]_{myo}',
    '@M_19@': r'#f([AA-food],$\alpha_{19}$,$\beta_{19}$,$\gamma_{19}$,$\delta_{19}$)# * [AA]_{myo}',
    '@M_20@': r'$m_{20}$ * [AA-food]',
    '@M_21@': r'$m_{21}$ * [Pyr]_{myo} * [[H]n-c]_{myo}',
    '@M_22@': r'$m_{22}$ * [Lac]_{myo}',
    '@M_23@': r'#f([INS],$\alpha_{23}$,$\beta_{23}$,$\gamma_{23}$,$\delta_{23}$)#* [Gluc-plasma]',
    '@M_24@': r'$m_{24}$ * [Cytr]_{myo}',
    '@M_25@': r'$m_{25}$ * [[H]nf-m]_{myo}',
    '@M_26@': r'$m_{26}$ * [ATP-m]_{myo}',
    '@M_27@': r'$m_{27}$ * [G3P]_{myo} * [[H]n-c]_{myo}',
    '@M_28@': r'$m_{28}$ * [Gly3P]_{myo}',
    '@M_29@': r'$m_{29}$ * [[H]f-m]_{myo}',
    '@M_30@': r'$m_{30}$ * [ATP-c]_{myo}',
    '@M_31@': r'#f([INS],$\alpha_{31}$,$\beta_{31}$,$\gamma_{31}$,$\delta_{31}$)#*[LD]_{myo}'
}
des_str_ = {
    # для нутриентов
    'D_{carb}': r'($a_{carb}$ * #F_{carb}(t)# - D_{carb}) / $\tau_{carb}$',
    'J_{carb}': r'(D_{carb} - J_{carb}) / $\tau_{carb}$',
    'D_{fat}': r'($a_{fat}$ * #F_{fat}(t)# - D_{fat}) / $\tau_{fat}$',
    'J_{fat}': r'(D_{fat} - J_{fat}) / $\tau_{fat}$',
    'D_{prot}': r'($a_{prot}$ * #F_{prot}(t)# - D_{prot}) / $\tau_{prot}$',
    'J_{prot}': r'(D_{prot} - J_{prot}) / $\tau_{prot}$',
    #################
    '[INS]': r'$\alpha$ * J_{carb} + $\beta$ * J_{fat} + $\gamma$ * J_{prot} - $CL$ * [INS]',
    '[Gluc-plasma]': r'J_{carb} - @M_23@',
    '[AA-food]': r'J_{prot} - @M_20@ + @M_19@',
    '[TAG-plasma]': r'J_{fat} - @M_0@',
    # блок 1
    '[FFA^{loc}_{pl}]_{myo}': r'@M_0@ - @M_1@ - #[SP]_{myo}(t)#',
    '[FFA]_{myo}': r'@M_1@ + 3 * @M_31@ - @M_2@',
    '[FA-CoA]_{myo}': r'@M_2@ - @M_3@ - @M_5@',
    '[TAG]_{myo}': r'@M_5@ - @M_6@',
    '[LD]_{myo}': r'@M_6@ - @M_31@',
    # блок 2
    '[Gly3P]_{myo}': r'@M_27@ - @M_5@ - @M_28@',
    '[G3P]_{myo}': r'@M_28@ + @M_8@ - @M_27@ - @M_11@',
    '[G6P]_{myo}': r'@M_7@ + @M_10@ - @M_8@ - @M_9@',
    '[Gluc]_{myo}': r'@M_23@ - @M_7@',
    '[GG]_{myo}': r'@M_9@ - @M_10@',
    '[Pyr]_{myo}': r'@M_11@ + @M_15@ - @M_12@ - @M_21@',
    '[Lac]_{myo}': r'@M_21@ - @M_22@',
    '[[H]n-c]_{myo}': r'@M_11@ - @M_21@ - @M_27@',
    '[ATP-c]_{myo}': r'@M_11@ - @M_30@',
    '[[H]f-m]_{myo}': r'@M_28@ - @M_29@',
    # блок 3
    '[Ac-CoA]_{myo}': r'@M_12@ + @M_3@ + @M_14@ + @M_13@ - @M_4@',
    '[Cytr]_{myo}': r'@M_4@ - @M_24@',
    '[OAA]_{myo}': r'@M_16@ + @M_24@ - @M_4@',
    '[CO_2]_{myo}': r'@M_24@ + @M_12@',
    '[[H]nf-m]_{myo}': r'@M_24@ + @M_29@ - @M_25@',
    '[ATP-m]_{myo}': r'@M_25@ - @M_26@ - @M_30@',
    # блок 4
    '[AA]_{myo}': r'@M_20@ + @M_17@ - @M_14@ - @M_15@ - @M_16@ - @M_18@ - @M_19@',
    '[muscle]_{myo}': r'@M_18@ - @M_17@'
}
