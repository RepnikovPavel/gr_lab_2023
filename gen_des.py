import numpy as np
from de_config import KB_plasma
from de_config import F_carb
from de_config import F_fat
from de_config import F_prot
from de_config import SP_myo
from de_config import sigmoid


def F_vec(y_vec,t, param_vec):
	buffer = np.zeros(shape=(33,))
	buffer[0] = (param_vec[47]*F_carb(t) - y_vec[0])/param_vec[44]
	buffer[1] = (y_vec[0] - y_vec[1])/param_vec[44]
	buffer[2] = (param_vec[48]*F_fat(t) - y_vec[2])/param_vec[45]
	buffer[3] = (y_vec[2] - y_vec[3])/param_vec[45]
	buffer[4] = (param_vec[49]*F_prot(t) - y_vec[4])/param_vec[46]
	buffer[5] = (y_vec[4] - y_vec[5])/param_vec[46]
	buffer[6] = -param_vec[0]*y_vec[6] + (param_vec[33]*y_vec[5] + (param_vec[12]*y_vec[3] + param_vec[1]*y_vec[1]))
	buffer[7] = -sigmoid(y_vec[6],param_vec[7],param_vec[18],param_vec[39],param_vec[28])*y_vec[7] + y_vec[1]
	buffer[8] = sigmoid(y_vec[8],param_vec[5],param_vec[16],param_vec[37],param_vec[26])*y_vec[31] - (param_vec[58]*y_vec[8] + y_vec[5])
	buffer[9] = -param_vec[50]*y_vec[9] + y_vec[3]
	buffer[10] = -SP_myo(t) + (param_vec[50]*y_vec[9] - sigmoid(y_vec[6],param_vec[6],param_vec[17],param_vec[38],param_vec[27])*y_vec[10])
	buffer[11] = -param_vec[67]*y_vec[11] + (sigmoid(y_vec[6],param_vec[6],param_vec[17],param_vec[38],param_vec[27])*y_vec[10] + (3*sigmoid(y_vec[6],param_vec[8],param_vec[19],param_vec[40],param_vec[29]))*y_vec[14])
	buffer[12] = -y_vec[15]*param_vec[71]*y_vec[12] + (param_vec[67]*y_vec[11] - param_vec[69]*y_vec[12])
	buffer[13] = -param_vec[72]*y_vec[13] + y_vec[15]*(param_vec[71]*y_vec[12])
	buffer[14] = param_vec[72]*y_vec[13] - sigmoid(y_vec[6],param_vec[8],param_vec[19],param_vec[40],param_vec[29])*y_vec[14]
	buffer[15] = -param_vec[65]*y_vec[15] - (y_vec[15]*param_vec[71]*y_vec[12] + y_vec[22]*(param_vec[64]*y_vec[16]))
	buffer[16] = -y_vec[23]*y_vec[22]*(param_vec[52]*y_vec[16]) - (y_vec[22]*param_vec[64]*y_vec[16] + (param_vec[65]*y_vec[15] + sigmoid(y_vec[6],param_vec[10],param_vec[21],param_vec[42],param_vec[31])*y_vec[17]))
	buffer[17] = -sigmoid(y_vec[6],param_vec[9],param_vec[20],param_vec[41],param_vec[30])*y_vec[17] - (sigmoid(y_vec[6],param_vec[10],param_vec[21],param_vec[42],param_vec[31])*y_vec[17] + (param_vec[51]*y_vec[19] + sigmoid(y_vec[6],param_vec[11],param_vec[22],param_vec[43],param_vec[32])*y_vec[18]))
	buffer[18] = sigmoid(y_vec[6],param_vec[7],param_vec[18],param_vec[39],param_vec[28])*y_vec[7] - sigmoid(y_vec[6],param_vec[9],param_vec[20],param_vec[41],param_vec[30])*y_vec[18]
	buffer[19] = -param_vec[51]*y_vec[19] + sigmoid(y_vec[6],param_vec[11],param_vec[22],param_vec[43],param_vec[32])*y_vec[17]
	buffer[20] = -y_vec[22]*param_vec[59]*y_vec[20] - (y_vec[20]*sigmoid(y_vec[6],param_vec[3],param_vec[14],param_vec[35],param_vec[24])*sigmoid(y_vec[25],param_vec[2],param_vec[13],param_vec[34],param_vec[23]) + (param_vec[55]*y_vec[31] + y_vec[23]*(y_vec[22]*(param_vec[52]*y_vec[16]))))
	buffer[21] = -param_vec[60]*y_vec[21] + y_vec[22]*(param_vec[59]*y_vec[20])
	buffer[22] = -y_vec[22]*param_vec[64]*y_vec[16] - (y_vec[22]*param_vec[59]*y_vec[20] + y_vec[23]*(y_vec[22]*(param_vec[52]*y_vec[16])))
	buffer[23] = -param_vec[68]*y_vec[23] + y_vec[23]*(y_vec[22]*(param_vec[52]*y_vec[16]))
	buffer[24] = param_vec[65]*y_vec[15] - param_vec[66]*y_vec[24]
	buffer[25] = -param_vec[70]*y_vec[25] + (param_vec[53]*KB_plasma(t) + (param_vec[54]*y_vec[31] + (param_vec[69]*y_vec[12] + y_vec[20]*(sigmoid(y_vec[6],param_vec[3],param_vec[14],param_vec[35],param_vec[24])*sigmoid(y_vec[25],param_vec[2],param_vec[13],param_vec[34],param_vec[23])))))
	buffer[26] = -param_vec[61]*y_vec[26] + param_vec[70]*y_vec[25]
	buffer[27] = -param_vec[70]*y_vec[25] + (param_vec[56]*y_vec[31] + param_vec[61]*y_vec[26])
	buffer[28] = param_vec[61]*y_vec[26] + y_vec[20]*(sigmoid(y_vec[6],param_vec[3],param_vec[14],param_vec[35],param_vec[24])*sigmoid(y_vec[25],param_vec[2],param_vec[13],param_vec[34],param_vec[23]))
	buffer[29] = -param_vec[62]*y_vec[29] + (param_vec[61]*y_vec[26] + param_vec[66]*y_vec[24])
	buffer[30] = -param_vec[68]*y_vec[23] + (param_vec[62]*y_vec[29] - param_vec[63]*y_vec[30])
	buffer[31] = -sigmoid(y_vec[8],param_vec[4],param_vec[15],param_vec[36],param_vec[25])*y_vec[31] - (param_vec[57]*y_vec[31] - (param_vec[56]*y_vec[31] - (param_vec[55]*y_vec[31] - (param_vec[54]*y_vec[31] + (param_vec[58]*y_vec[8] + sigmoid(y_vec[8],param_vec[5],param_vec[16],param_vec[37],param_vec[26])*y_vec[32])))))
	buffer[32] = param_vec[57]*y_vec[31] - sigmoid(y_vec[8],param_vec[4],param_vec[15],param_vec[36],param_vec[25])*y_vec[32]
	return buffer