import numpy as np
from numba import jit
from scipy.integrate import odeint

# BM Bone mineral mass in g
# BW Body weight in g
# CarbOx Rate of carbohydrate oxidation in kcal/day
# CI Carbohydrate intake rate in kcal/day
# DF Rate of endogenous lipolysis in g/day
# DG Rate of glycogenolysis in g/day
# DNL Rate of de novo lipogenesis in kcal/day
# DP Rate of proteolysis in g/day
# ECF Extracellular fluid mass in g
# ECP Extracellular protein mass in g
# EI Energy intake in kcal/day
# FM Body fat mass in g
# FatOx Rate of fat oxidation in kcal/day
# fC Carbohydrate oxidation fraction
# fF Fat oxidation fraction
# FFM Fat-free body mass in g
# FI Fat intake rate in kcal/day
# fP Protein oxidation fraction
# G Body glycogen mass in g
# G3P Rate of glycerol 3-phosphate synthesis in kcal/day
# GNGF Rate of gluconeogenesis from glycerol in kcal/day
# GNGP Rate of gluconeogenesis from protein in kcal/day
# ICS Intracellular solid mass in g
# ICW Intracellular water mass in g
# KetOx Rate of ketone oxidation in kcal/day
# KTG Rate of ketogenesis in kcal/day
# KUexcr Rate of ketone excretion in kcal/day
# LCM Lean tissue cell mass in g
# Nexcr Nitrogen excretion rate in g/day
# NPRQ Nonprotein respiratory quotient
# P Intracellular protein mass in g
# PAE Physical activity energy expenditure in kcal/day
# PI Protein intake rate in kcal/day
# ProtOx Rate of protein oxidation in kcal/day
# RMR Resting metabolic rate in kcal/day
# RQ Respiratory quotient
# SynthF Rate of fat synthesis in g/day
# SynthG Rate of glycogen synthesis in g/day
# SynthP Rate of protein synthesis in g/day
# T Adaptive thermogenesis
# TEE Total energy expenditure in kcal/day
# TEF Thermic effect of feeding in kcal/day
# TG Triacylglyceride
# V˙ CO2 Rate of carbon dioxide production in liters/day
# V˙ O2 Rate of oxygen consumption in liters/day


# \(BM\)
# \(BW\)
# \(CarbOx\)
# \(CI\)
# \(D_{F}\)
# \(D_{G}\)
# \(DNL\)
# \(D_{P}\)
# \(ECF\)
# \(ECP\)
# \(EI\)
# \(FM\)
# \(FatOx\)
# \(f_{C}\)
# \(f_{F}\)
# \(FFM\)
# \(FI\)
# \(f_{P}\)
# \(G\)
# \(G3P\)
# \(GNG_{F}\)
# \(GNG_{P}\)
# \(ICS\)
# \(ICW\)
# \(KetOx\)
# \(KTG\)
# \(KU_{excr}\)
# \(LCM\)
# \(N_{excr}\)
# \(NPRQ\)
# \(P\)
# \(PAE\)
# \(PI\)
# \(ProtOx\)
# \(RMR\)
# \(RQ\)
# \(Synth_{F}\)
# \(Synth_{G}\)
# \(Synth_{P}\)
# \(T\)
# \(TEE\)
# \(TEF\)
# \(TG\)
# \(\dot{V}_{CO_{2}}\)
# \(\dot{V}_{O_{2}}\)

# BM Bone mineral mass in g
# BW Body weight in g
# CarbOx Rate of carbohydrate oxidation in kcal/day
# CI Carbohydrate intake rate in kcal/day
# DF Rate of endogenous lipolysis in g/day
# DG Rate of glycogenolysis in g/day
# DNL Rate of de novo lipogenesis in kcal/day
# DP Rate of proteolysis in g/day
# ECF Extracellular fluid mass in g
# ECP Extracellular protein mass in g
# EI Energy intake in kcal/day
# FM Body fat mass in g
# FatOx Rate of fat oxidation in kcal/day
# fC Carbohydrate oxidation fraction
# fF Fat oxidation fraction
# FFM Fat-free body mass in g
# FI Fat intake rate in kcal/day
# fP Protein oxidation fraction
# G Body glycogen mass in g
# G3P Rate of glycerol 3-phosphate synthesis in kcal/day
# GNGF Rate of gluconeogenesis from glycerol in kcal/day
# GNGP Rate of gluconeogenesis from protein in kcal/day
# ICS Intracellular solid mass in g
# ICW Intracellular water mass in g
# KetOx Rate of ketone oxidation in kcal/day
# KTG Rate of ketogenesis in kcal/day
# KUexcr Rate of ketone excretion in kcal/day
# LCM Lean tissue cell mass in g
# Nexcr Nitrogen excretion rate in g/day
# NPRQ Nonprotein respiratory quotient
# P Intracellular protein mass in g
# PAE Physical activity energy expenditure in kcal/day
# PI Protein intake rate in kcal/day
# ProtOx Rate of protein oxidation in kcal/day
# RMR Resting metabolic rate in kcal/day
# RQ Respiratory quotient
# SynthF Rate of fat synthesis in g/day
# SynthG Rate of glycogen synthesis in g/day
# SynthP Rate of protein synthesis in g/day
# T Adaptive thermogenesis
# TEE Total energy expenditure in kcal/day
# TEF Thermic effect of feeding in kcal/day
# TG Triacylglyceride
# V˙ CO2 Rate of carbon dioxide production in liters/day
# V˙ O2 Rate of oxygen consumption in liters/day

def rho_C(t:float):
    return 1.0
def rho_F(t:float):
    return 1.0
def rho_P(t:float):
    return 1.0
def CI(t:float):
    return 1.0
def DNL(t:float):
    return 1.0
def GNG_P(t:float):
    return 1.0
def GNG_F(t:float):
    return 1.0
def G3P(t:float):
    return 1.0
def CarbOx(t:float):
    return 1.0
def M_FFA(t:float):
    return 1.0
def M_TG(t:float):
    return 1.0
def eps_d(t:float):
    return 1.0
def DNL(t:float):
    return 1.0
def KU_excr(t:float):
    return 1.0
def eps_k(t:float):
    return 1.0
def KTG(t:float):
    return 1.0
def FatOx(t:float):
    return 1.0
def PI(t:float):
    return 1.0
def GNG_P(t:float):
    return 1.0
def ProtOx(t:float):
    return 1.0

def tau_L(t:float):
    return 1.0
def K_SL_L(t:float):
    return 1.0
def A_L(t:float):
    return 1.0
def B_L(t:float):
    return 1.0
def k_L(t:float):
    return 1.0
def F_keys(t:float):
    return 1.0
def S_L(t:float):
    return 1.0

def Na(t:float):
    return 1.0
def deltaNa_diet(t:float):
    return 1.0
def xi_Na(t:float):
    return 1.0
def ECF_init(t:float):
    return 1.0
def xi_CI(t:float):
    return 1.0
def CI_b(t:float):
    return 1.0
def tau_BW(t:float):
    return 1.0
def xi_BW(t:float):
    return 1.0
def BW(t:float):
    return 1.0
def BW_init(t:float):
    return 1.0

def GFPL_diet_ODEsystem(y_vec: np.array,t: float, param_vec: np.array):
    buffer = np.zeros(shape=(4,))
    G_ = y_vec[0]
    F_ = y_vec[1]
    P_ = y_vec[2]
    L_diet_ = y_vec[3]
    dGdt_t = (1/rho_C(t))*(CI(t)-DNL(t)+GNG_P(t)+GNG_F(t)-G3P(t)-CarbOx(t))
    dFdt_t = (1/rho_F(t))*(3*M_FFA(t)/M_TG(t) + eps_d(t)*DNL(t) - KU_excr(t)- (1-eps_k(t))*KTG(t)-FatOx(t))
    dPdt_t = (1/rho_P(t))*(PI(t)-GNG_P(t)-ProtOx(t))
    dL_dietdt_t = (1/tau_L(t))*(((K_SL_L(t)*(1.0+(A_L(t)-B_L(t))*np.exp(-k_L(t)*(CI(t)/CI_b(t)))+B_L(t)))/(K_SL_L(t)+np.maximum(0.0,np.power(F_/F_keys(t)-1,S_L(t)))))-L_diet_)
    buffer[0]= dGdt_t
    buffer[1]= dFdt_t
    buffer[2]= dPdt_t
    buffer[3]= dL_dietdt_t
    return buffer
G_start = 1.0
F_start = 1.0
P_start = 1.0
L_diet_start = 1.0

def solve_GFPL_diet_ODEsystem():
    tau = 0.1  # [min] 
    t_0 = 0.0  # [min]
    t_end = 1440.0 # [min]
    N = int((t_end-t_0)/tau)+1
    time_grid = np.linspace(start=t_0, stop=t_end, num=N)
    start_point = np.array([G_start,F_start,P_start,L_diet_start])
    solutions = odeint(func=GFPL_diet_ODEsystem, y0=start_point, t=time_grid, args=((),), full_output=False)

    def G_t_(t:float):
        if t< t_0 or t> t_end:
            return np.nan
        g_arr = solutions[:,0]
        index_ = int(t/tau)
        return g_arr[index_]

    def dGdt_t_(t:float):
        if t< t_0 or t> t_end:
            return np.nan
        g_arr = solutions[:,0]
        index_ = int(t/tau)
        if index_ == 0:
            return (g_arr[1]-g_arr[0])/tau
        elif index_ == (N-1):
            return (g_arr[-1]-g_arr[-2])/tau
        else:
            return (g_arr[index_+1]-g_arr[index_-1])/(2*tau)

    def F_t_(t:float):
        if t< t_0 or t> t_end:
            return np.nan
        f_arr = solutions[:,1]
        index_ = int(t/tau)
        return f_arr[index_]

    def dFdt_t_(t:float):
        if t< t_0 or t> t_end:
            return np.nan
        f_arr = solutions[:,1]
        index_ = int(t/tau)
        if index_ == 0:
            return (f_arr[1]-f_arr[0])/tau
        elif index_ == (N-1):
            return (f_arr[-1]-f_arr[-2])/tau
        else:
            return (f_arr[index_+1]-f_arr[index_-1])/(2*tau)
        
    def P_t_(t:float):
        if t< t_0 or t> t_end:
            return np.nan
        p_arr = solutions[:,2]
        index_ = int(t/tau)
        return p_arr[index_]

    def dPdt_t_(t:float):
        if t< t_0 or t> t_end:
            return np.nan
        p_arr = solutions[:,2]
        index_ = int(t/tau)
        if index_ == 0:
            return (p_arr[1]-p_arr[0])/tau
        elif index_ == (N-1):
            return (p_arr[-1]-p_arr[-2])/tau
        else:
            return (p_arr[index_+1]-p_arr[index_-1])/(2*tau)
        
    def L_diet_t_(t:float):
        if t< t_0 or t> t_end:
            return np.nan
        l_arr = solutions[:,3]
        index_ = int(t/tau)
        return l_arr[index_]

    def dL_dietdt_t_(t:float):
        if t< t_0 or t> t_end:
            return np.nan
        l_arr = solutions[:,3]
        index_ = int(t/tau)
        if index_ == 0:
            return (l_arr[1]-l_arr[0])/tau
        elif index_ == (N-1):
            return (l_arr[-1]-l_arr[-2])/tau
        else:
            return (l_arr[index_+1]-l_arr[index_-1])/(2*tau)

    return G_t_, dGdt_t_,F_t_,dFdt_t_,P_t_,dPdt_t_,L_diet_t_,dL_dietdt_t_


G,dGdt,F,dFdt,P,dPdt,L_diet,dL_dietdt = solve_GFPL_diet_ODEsystem()


def ECF_ODEsystem(y_vec: np.array,t: float, param_vec: np.array):
    buffer = np.zeros(shape=(2,))
    ECF_ = y_vec[0]
    delta_ECF_ = y_vec[1]
    dECFdt_t = (1/Na(t))*(deltaNa_diet(t)-xi_Na(t)*(ECF_-ECF_init(t))-xi_CI(t)*(1-CI(t)/CI_b(t))) +delta_ECF_
    ddeltaECFdt_t = (1/tau_BW(t))*((BW(t)-BW_init(t)) - delta_ECF_)
    buffer[0] = dECFdt_t
    buffer[1] = ddeltaECFdt_t
    return buffer

ECF_start = 1.0
deltaECF_start = 1.0

def solve_ECF_ODEsystem():
    tau = 0.1  # [min] 
    t_0 = 0.0  # [min]
    t_end = 1440.0 # [min]
    N = int((t_end-t_0)/tau)+1
    time_grid = np.linspace(start=t_0, stop=t_end, num=N)
    start_point = np.array([ECF_start, deltaECF_start])
    solutions = odeint(func=ECF_ODEsystem, y0=start_point, t=time_grid, args=((),), full_output=False)

    def ECF_t_(t:float):
        if t< t_0 or t> t_end:
            return np.nan
        g_arr = solutions[:,0]
        index_ = int(t/tau)
        return g_arr[index_]

    def dECFdt_t_(t:float):
        if t< t_0 or t> t_end:
            return np.nan
        g_arr = solutions[:,0]
        index_ = int(t/tau)
        if index_ == 0:
            return (g_arr[1]-g_arr[0])/tau
        elif index_ == (N-1):
            return (g_arr[-1]-g_arr[-2])/tau
        else:
            return (g_arr[index_+1]-g_arr[index_-1])/(2*tau)
        
    def deltaECF_t_(t:float):
        if t< t_0 or t> t_end:
            return np.nan
        f_arr = solutions[:,1]
        index_ = int(t/tau)
        return f_arr[index_]

    def ddeltaECFdt_t_(t:float):
        if t< t_0 or t> t_end:
            return np.nan
        f_arr = solutions[:,1]
        index_ = int(t/tau)
        if index_ == 0:
            return (f_arr[1]-f_arr[0])/tau
        elif index_ == (N-1):
            return (f_arr[-1]-f_arr[-2])/tau
        else:
            return (f_arr[index_+1]-f_arr[index_-1])/(2*tau)
    return ECF_t_,dECFdt_t_,deltaECF_t_,ddeltaECFdt_t_

ECF,dECFdt,deltaECF,ddeltaECFdt = solve_ECF_ODEsystem()



def tau_T(t:float):
    return 1.0
def lambda_1(t:float):
    return 1.0
def lambda_2(t:float):
    return 1.0
def EI(t:float):
    return 1.0
def EI_b(t:float):
    return 1.0
def deltaEI(t:float):
    return 1.0

def T_ODEsystem(y_vec: np.array,t: float, param_vec: np.array):
    buffer = np.zeros(shape=(1,))
    T = y_vec[0]
    dTdt_t = 0.0
    if EI(t) < EI_b(t):
        dTdt_t = (1/tau_T(t))*(lambda_1(t)*(deltaEI(t)/EI_b(t))-T)
    else:
        dTdt_t = (1/tau_T(t))*(lambda_2(t)*(deltaEI(t)/EI_b(t))-T)
    buffer[0] = dTdt_t 
    return buffer

T_start = 

def solve_T_ODEsystem():
    tau = 0.1  # [min] 
    t_0 = 0.0  # [min]
    t_end = 1440.0 # [min]
    N = int((t_end-t_0)/tau)+1
    time_grid = np.linspace(start=t_0, stop=t_end, num=N)
    start_point = np.array([ECF_start, deltaECF_start])
    solutions = odeint(func=ECF_ODEsystem, y0=start_point, t=time_grid, args=((),), full_output=False)

    def ECF_t_(t:float):
        if t< t_0 or t> t_end:
            return np.nan
        g_arr = solutions[:,0]
        index_ = int(t/tau)
        return g_arr[index_]

    def dECFdt_t_(t:float):
        if t< t_0 or t> t_end:
            return np.nan
        g_arr = solutions[:,0]
        index_ = int(t/tau)
        if index_ == 0:
            return (g_arr[1]-g_arr[0])/tau
        elif index_ == (N-1):
            return (g_arr[-1]-g_arr[-2])/tau
        else:
            return (g_arr[index_+1]-g_arr[index_-1])/(2*tau)



def FFM(t:float):
    return 1.0





    