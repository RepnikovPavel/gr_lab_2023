M_str = {
    'M_0':  'm_0 * TAG-plasma',
    'M_1':  'f([INS],α_1,β_1,γ_1,δ_1) * FFA-plasma-local-m',
    'M_2':  'm_2 * FFA_m',
    'M_3':  'm_3 * FA-CoA_m',
    'M_4':  'm_4 * Ac-CoA_m',
    'M_5':  'm_5 * FA-CoA_m * Gly3P_m',
    'M_6':  'm_6*TAG_m',
    'M_7':  'f([INS],α_7,β_7,γ_7,δ_7) * Gluc_m',
    'M_8':  'f([INS],α_8,β_8,γ_8,δ_8) * G6P_m',
    'M_9':  'f([INS],α_9,β_9,γ_9,δ_9) * G6P_m',
    'M_10': 'm_10  * GG_m',
    'M_11': 'm_11*G3P_m*[H]n_c_m*ATP_c_m',
    'M_12': 'f([INS],α_12,β_12,γ_12,δ_12)*f([Ac_Coa],α_12_2,β_12_2,γ_12_2,δ_12_2)*Pyr_m',
    'M_13': 'm_13*KB_plasma',
    'M_14': 'm_14*AA_m',
    'M_15': 'm_15*AA_m',
    'M_16': 'm_16*AA_m',
    'M_17': 'f(AA_food,α_17,β_17,γ_17,δ_17) * muscle',
    'M_18': 'm_18*AA_m',
    'M_19': 'f(AA_food,α_19,β_19,γ_19,δ_19) * AA_m',
    'M_20': 'm_20*AA-food',
    'M_21': 'm_21*Pyr_m*[H]n_c_m',
    'M_22': 'm_22*Lac_m',
    'M_23': 'f([INS],α_23,β_23,γ_23,δ_23)* Cluc-plasma',
    'M_24': 'm_24*Citr_m',
    'M_25': 'm_25*[H]nf_m_myo',
    'M_26': 'm_26*ATP_m_myo',
    'M_27': 'm_27*G3P_m*[H]n_c_m',
    'M_28': 'm_28*Gly3P_m',
    'M_29': 'm_29*[H]f_m_myo',
    'M_30': 'm_30*ATP_c_m',
    'M_31': 'f([INS],α_31,β_31,γ_31,δ_31)*LD_m'
}
des_str_ = {
    '[Gluc-plasma]':            'J_carb - M_23',
    '[AA-food]':                'J_prot - M_20 + M_19',
    '[TAG-plasma]':             'J_fat - M_0',
    # блок 1
    '[FFA^{loc}_{pl}]_{myo}':   'M_0-M_1-SP_m',
    '[FFA]_{myo}':              'M_1+3*M_31-M_2',
    '[FA-CoA]_{myo}':           'M_2-M_3-M_5',
    '[TAG]_{myo}':              'M_5-M_6',
    '[LD]_{myo}':               'M_6-M_31',
    # блок 2
    '':'',
    '':'',
    '':'',
    '':'',



}
F_str = {

}
Y_str = {
}
