free_functions = ['f']
external_links = ['[KB-plasma]']

aliases_ = {
    '@M_0@': r'm_0 * [TAG-plasma]',
    '@M_1@': r'f([INS],\alpha_1,\beta_1,\gamma_1,\delta_1) * [FFA^{loc}_{pl}]_{myo}',
    '@M_2@': r'm_2 * [FFA]_{myo}',
    '@M_3@': r'm_3 * [FA-CoA]_{myo}',
    '@M_4@': r'm_4 * [Ac-CoA]_{myo}',
    '@M_5@': r'm_5 * [FA-CoA]_{myo} * [Gly3P]_{myo}',
    '@M_6@': r'm_6 * [TAG]_{myo}',
    '@M_7@': r'f([INS],\alpha_7,\beta_7,\gamma_7,\delta_7) * [Gluc]_{myo}',
    '@M_8@': r'f([INS],\alpha_8,\beta_8,\gamma_8,\delta_8) * [G6P]_{myo}',
    '@M_9@': r'f([INS],\alpha_9,\beta_9,\gamma_9,\delta_9) * [G6P]_{myo}',
    '@M_10@': r'm_10 * [GG]_{myo}',
    '@M_11@': r'm_11 * [G3P]_{myo} * [[H]n-c]_{myo} * [ATP-c]_{myo}',
    '@M_12@': r'f([INS],\alpha_12,\beta_12,\gamma_12,\delta_12)*f([Ac-CoA]_{myo},alpha_12_2,beta_12_2,gamma_12_2, '
              r'delta_12_2)*[Pyr]_{myo}',
    '@M_13@': r'm_13 * [KB-plasma]',
    '@M_14@': r'm_14 * [AA]_{myo}',
    '@M_15@': r'm_15 * [AA]_{myo}',
    '@M_16@': r'm_16 * [AA]_{myo}',
    '@M_17@': r'f([AA-food],\alpha_17,\beta_17,\gamma_17,\delta_17) * [muscle]_{myo}',
    '@M_18@': r'm_18 * [AA]_{myo}',
    '@M_19@': r'f([AA-food],\alpha_19,\beta_19,\gamma_19,\delta_19) * [AA]_{myo}',
    '@M_20@': r'm_20 * [AA-food]',
    '@M_21@': r'm_21 * [Pyr]_{myo} * [[H]n-c]_{myo}',
    '@M_22@': r'm_22 * [Lac]_{myo}',
    '@M_23@': r'f([INS],\alpha_23,\beta_23,\gamma_23,\delta_23)* [Gluc-plasma]',
    '@M_24@': r'm_24 * [Cytr]_{myo}',
    '@M_25@': r'm_25 * [[H]nf-m]_{myo}',
    '@M_26@': r'm_26 * [ATP-m]_{myo}',
    '@M_27@': r'm_27 * [G3P]_{myo} * [[H]n-c]_{myo}',
    '@M_28@': r'm_28 * [Gly3P]_{myo}',
    '@M_29@': r'm_29 * [[H]f-m]_{myo}',
    '@M_30@': r'm_30 * [ATP-c]_{myo}',
    '@M_31@': r'f([INS],\alpha_31,\beta_31,\gamma_31,\delta_31)*[LD]_{myo}'
}
des_str_ = {
    # для нутриентов
    'D_carb': r'(a_carb * F_carb - D_carb) / tau_carb',
    'J_carb': r'(D_carb - J_carb) / tau_carb',
    'D_fat': r'(a_fat * F_fat - D_fat) / tau_fat',
    'J_fat': r'(D_fat - J_fat) / tau_fat',
    'D_prot': r'(a_prot * F_prot - D_prot) / tau_prot',
    'J_prot': r'(D_prot - J_prot) / tau_prot',
    #################
    '[INS]': r'\alpha * J_carb + \beta * J_fat + \gamma * J_prot - CL * [INS]',
    '[Gluc-plasma]': r'J_carb - @M_23@',
    '[AA-food]': r'J_prot - @M_20@ + @M_19@',
    '[TAG-plasma]': r'J_fat - @M_0@',
    # блок 1
    '[FFA^{loc}_{pl}]_{myo}': r'@M_0@ - @M_1@ - [SP]_{myo}',
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
