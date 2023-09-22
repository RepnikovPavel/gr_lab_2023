import numpy as np
from numba import jit
from sysv1_de_config import F_carb
from sysv1_de_config import F_fat
from sysv1_de_config import F_prot
from sysv1_de_config import sigmoid


@jit(nopython=True, cache=True)
def F_vec(y_vec: np.array,t: float, param_vec: np.array):
	buffer = np.zeros(shape=(53,))
	buffer[0] = sigmoid(y_vec[52])*(y_vec[6]*(param_vec[23]*y_vec[3])) - sigmoid(G*(L*N))*sigmoid(C*(A*M))*(param_vec[19]*y_vec[0])
	buffer[1] = -param_vec[16]*y_vec[1] - (param_vec[15]*y_vec[1] - (param_vec[14]*y_vec[1] + param_vec[17]*y_vec[38]))
	buffer[2] = -param_vec[22]*y_vec[2] + (sigmoid(y_vec[52])*(param_vec[20]*y_vec[37]) - sigmoid(y_vec[52])*param_vec[21]*y_vec[2])
	buffer[3] = -param_vec[24]*y_vec[3] - (sigmoid(y_vec[52])*y_vec[6]*(param_vec[23]*y_vec[3]) + (sigmoid(G*(L*N))*(param_vec[25]*y_vec[8]) + (param_vec[22]*y_vec[2] + sigmoid(y_vec[52])*((2*param_vec[21])*y_vec[2]))))
	buffer[4] = -sigmoid(y_vec[52])*param_vec[8]*y_vec[4] - (sigmoid(G*(L*N))*param_vec[7]*y_vec[4] + (param_vec[16]*y_vec[1] + (param_vec[24]*y_vec[3] + param_vec[9]*y_vec[8])))
	buffer[5] = -y_vec[7]*param_vec[13]*y_vec[8] - (y_vec[9]*(8*param_vec[10])*y_vec[5] + (param_vec[15]*y_vec[1] + (sigmoid(y_vec[52])*(param_vec[7]*y_vec[4]) + sigmoid(y_vec[52])*(param_vec[11]*y_vec[7]))))
	buffer[6] = -sigmoid(y_vec[52])*y_vec[6]*((3*param_vec[23])*y_vec[3]) + (sigmoid(y_vec[52])*(param_vec[18]*y_vec[41]) + y_vec[9]*(param_vec[10]*y_vec[5]))
	buffer[7] = -param_vec[12]*y_vec[7] - (sigmoid(y_vec[52])*param_vec[11]*y_vec[7] + y_vec[7]*(param_vec[13]*y_vec[8]))
	buffer[8] = -y_vec[7]*param_vec[13]*y_vec[8] - (param_vec[9]*y_vec[8] - (sigmoid(G*(L*N))*param_vec[25]*y_vec[8] + (param_vec[14]*y_vec[1] + (param_vec[12]*y_vec[7] + (sigmoid(y_vec[52])*(param_vec[8]*y_vec[4]) + sigmoid(G*(L*N))*(param_vec[11]*y_vec[7]))))))
	buffer[9] = -y_vec[9]*(14*param_vec[10])*y_vec[5] + (param_vec[22]*y_vec[2] + param_vec[9]*y_vec[8])
	buffer[10] = sigmoid(y_vec[52])*(param_vec[29]*y_vec[11]) - sigmoid(G*(L*N))*sigmoid(C*(A*M))*(param_vec[30]*y_vec[10])
	buffer[11] = -param_vec[33]*y_vec[11] - (sigmoid(y_vec[52])*param_vec[31]*y_vec[11] - (sigmoid(G*(L*N))*param_vec[29]*y_vec[11] - (sigmoid(C*(A*M))*param_vec[50]*y_vec[11] + (param_vec[32]*y_vec[12] + (sigmoid(G*(L*N))*(param_vec[51]*y_vec[37]) + sigmoid(y_vec[52])*(sigmoid(y_vec[52])*(param_vec[30]*y_vec[10])))))))
	buffer[12] = -sigmoid(y_vec[52])*y_vec[15]*(param_vec[40]*y_vec[12]) - (param_vec[34]*y_vec[12] - (2*param_vec[32]*y_vec[12] + (sigmoid(G*(L*N))*(param_vec[43]*y_vec[18]) + (param_vec[33]*y_vec[11] + (param_vec[52]*y_vec[40] + sigmoid(y_vec[52])*((2*param_vec[31])*y_vec[11]))))))
	buffer[13] = -sigmoid(y_vec[52])*sigmoid(y_vec[52])*(param_vec[45]*y_vec[13]) - (sigmoid(G*(L*N))*param_vec[35]*y_vec[13] + (param_vec[49]*y_vec[20] + (sigmoid(C*(A*M))*(param_vec[44]*y_vec[18]) + (param_vec[34]*y_vec[12] + param_vec[53]*y_vec[42]))))
	buffer[14] = -sigmoid(y_vec[52])*y_vec[21]*((8*param_vec[38])*y_vec[14]) - (sigmoid(G*(L*N))*y_vec[21]*((3*param_vec[36])*y_vec[14]) + (param_vec[47]*y_vec[20] + (param_vec[46]*y_vec[20] + (sigmoid(y_vec[52])*(param_vec[35]*y_vec[13]) + sigmoid(y_vec[52])*((8*param_vec[37])*y_vec[15])))))
	buffer[15] = -sigmoid(y_vec[52])*y_vec[15]*((3*param_vec[40])*y_vec[12]) - (sigmoid(G*(L*N))*param_vec[37]*y_vec[15] + (param_vec[56]*y_vec[41] + sigmoid(y_vec[52])*(y_vec[21]*(param_vec[38]*y_vec[14]))))
	buffer[16] = -param_vec[57]*y_vec[16] + sigmoid(y_vec[52])*(y_vec[15]*(param_vec[40]*y_vec[12]))
	buffer[17] = sigmoid(y_vec[52])*(y_vec[21]*(param_vec[36]*y_vec[14])) - sigmoid(y_vec[52])*param_vec[55]*y_vec[17]
	buffer[18] = -sigmoid(G*(L*N))*param_vec[44]*y_vec[18] - (sigmoid(C*(A*M))*param_vec[43]*y_vec[18] - (y_vec[18]*param_vec[41]*y_vec[14] + (param_vec[48]*y_vec[20] + (param_vec[46]*y_vec[20] + (param_vec[42]*y_vec[19] + sigmoid(G*(L*N))*(sigmoid(y_vec[52])*(param_vec[45]*y_vec[13])))))))
	buffer[19] = -param_vec[46]*y_vec[20] - (param_vec[42]*y_vec[19] + y_vec[18]*(param_vec[41]*y_vec[14]))
	buffer[20] = -param_vec[49]*y_vec[20] - (param_vec[48]*y_vec[20] + (param_vec[39]*y_vec[38] - param_vec[47]*y_vec[20]))
	buffer[21] = -sigmoid(y_vec[52])*y_vec[21]*((14*param_vec[38])*y_vec[14]) + ((6*param_vec[33])*y_vec[11] + sigmoid(y_vec[52])*(param_vec[44]*y_vec[18]))
	buffer[22] = sigmoid(y_vec[52])*(param_vec[81]*y_vec[23]) - sigmoid(G*(L*N))*sigmoid(C*(A*M))*(param_vec[82]*y_vec[22])
	buffer[23] = -sigmoid(y_vec[52])*param_vec[83]*y_vec[23] - (sigmoid(G*(L*N))*param_vec[81]*y_vec[23] + (sigmoid(C*(A*M))*(param_vec[73]*y_vec[37]) + sigmoid(y_vec[52])*(sigmoid(y_vec[52])*(param_vec[82]*y_vec[22]))))
	buffer[24] = -param_vec[63]*y_vec[24] + sigmoid(y_vec[52])*((2*param_vec[83])*y_vec[23])
	buffer[25] = -param_vec[76]*y_vec[25] - (sigmoid(y_vec[52])*param_vec[64]*y_vec[25] + (param_vec[63]*y_vec[24] + param_vec[70]*y_vec[28]))
	buffer[26] = -y_vec[30]*param_vec[66]*y_vec[26] + (param_vec[71]*y_vec[28] + ((8*param_vec[65])*y_vec[27] + ((2*param_vec[77])*y_vec[43] + sigmoid(y_vec[52])*(param_vec[64]*y_vec[25]))))
	buffer[27] = -param_vec[65]*y_vec[27] + sigmoid(y_vec[52])*(param_vec[78]*y_vec[41])
	buffer[28] = param_vec[75]*(M*(u*(s*(c*(e*l))))) + param_vec[79]*y_vec[38]
	buffer[29] = -param_vec[67]*y_vec[29] + y_vec[30]*(param_vec[66]*y_vec[26])
	buffer[30] = -y_vec[30]*param_vec[66]*y_vec[26] + (param_vec[67]*y_vec[29] + param_vec[72]*y_vec[28])
	buffer[31] = -param_vec[68]*y_vec[31] + ((2*param_vec[63])*y_vec[24] - param_vec[76]*y_vec[25])
	buffer[32] = -param_vec[69]*y_vec[32] + (param_vec[68]*y_vec[31] + ((14*param_vec[65])*y_vec[27] + param_vec[77]*y_vec[43]))
	buffer[33] = (2*param_vec[67])*y_vec[29] + sigmoid(y_vec[52])*(param_vec[64]*y_vec[25])
	buffer[34] = param_vec[69]*y_vec[32]
	buffer[35] = param_vec[63]*y_vec[24]
	buffer[36] = (2*param_vec[69])*y_vec[32]
	buffer[37] = -param_vec[59]*y_vec[37] - (sigmoid(G*(L*N))*param_vec[20]*y_vec[37] - (sigmoid(y_vec[52])*param_vec[73]*y_vec[37] - (sigmoid(y_vec[52])*param_vec[51]*y_vec[37] + (sigmoid(y_vec[52])*(param_vec[50]*y_vec[11]) + y_vec[47]))))
	buffer[38] = -param_vec[79]*y_vec[38] - (param_vec[62]*y_vec[38] - (param_vec[39]*y_vec[38] - (param_vec[17]*y_vec[38] + (param_vec[80]*y_vec[28] + y_vec[51]))))
	buffer[39] = -sigmoid(y_vec[52])*param_vec[58]*y_vec[39] + (param_vec[57]*y_vec[16] + y_vec[49])
	buffer[40] = -param_vec[52]*y_vec[40] + (sigmoid(y_vec[52])*(param_vec[58]*y_vec[39]) + sigmoid(G*(L*N))*(sigmoid(C*(A*M))*(param_vec[19]*y_vec[0])))
	buffer[41] = -param_vec[61]*y_vec[41] - (sigmoid(y_vec[52])*param_vec[78]*y_vec[41] - (param_vec[56]*y_vec[41] - (sigmoid(G*(L*N))*param_vec[18]*y_vec[41] + (sigmoid(C*(A*M))*((3*param_vec[58])*y_vec[39]) + sigmoid(y_vec[52])*(sigmoid(y_vec[52])*((3*param_vec[19])*y_vec[0]))))))
	buffer[42] = -param_vec[53]*y_vec[42] + param_vec[76]*y_vec[25]
	buffer[43] = -param_vec[60]*y_vec[43] - (param_vec[77]*y_vec[43] + sigmoid(y_vec[52])*(param_vec[54]*y_vec[14]))
	buffer[44] = sigmoid(y_vec[52])*(param_vec[55]*y_vec[17])
	buffer[45] = param_vec[49]*y_vec[20] + (param_vec[48]*y_vec[20] + (param_vec[47]*y_vec[20] + (param_vec[72]*y_vec[28] + (param_vec[71]*y_vec[28] + (param_vec[70]*y_vec[28] + (param_vec[16]*y_vec[1] + (param_vec[15]*y_vec[1] + (param_vec[14]*y_vec[1] + param_vec[62]*y_vec[38]))))))))
	buffer[46] = (param_vec[26]*F_carb(t) - y_vec[46])/param_vec[4]
	buffer[47] = (y_vec[46] - y_vec[47])/param_vec[4]
	buffer[48] = (param_vec[27]*F_fat(t) - y_vec[48])/param_vec[5]
	buffer[49] = (y_vec[48] - y_vec[49])/param_vec[5]
	buffer[50] = (param_vec[28]*F_prot(t) - y_vec[50])/param_vec[6]
	buffer[51] = (y_vec[50] - y_vec[51])/param_vec[6]
	buffer[52] = -param_vec[0]*y_vec[52] + (param_vec[3]*y_vec[51] + (param_vec[1]*y_vec[47] + param_vec[2]*y_vec[49]))
	return buffer