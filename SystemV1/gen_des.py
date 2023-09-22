import numpy as np
from numba import jit
from sysv1_de_config import F_carb
from sysv1_de_config import F_fat
from sysv1_de_config import F_prot
from sysv1_de_config import sigmoid


@jit(nopython=True, cache=True)
def F_vec(y_vec: np.array,t: float, param_vec: np.array):
	buffer = np.zeros(shape=(59,))
	buffer[0] = sigmoid(y_vec[56],param_vec[8],param_vec[27],param_vec[64],param_vec[45])*(y_vec[7]*(param_vec[109]*y_vec[3])) - 1*sigmoid(6*G,param_vec[80])*(param_vec[105]*y_vec[0])
	buffer[1] = -param_vec[103]*y_vec[1] - (param_vec[102]*y_vec[1] - (param_vec[100]*y_vec[1] + param_vec[101]*y_vec[42]))
	buffer[2] = -param_vec[108]*y_vec[2] + (sigmoid(y_vec[56],param_vec[6],param_vec[25],param_vec[62],param_vec[43])*(param_vec[106]*y_vec[41]) - sigmoid(y_vec[56],param_vec[7],param_vec[26],param_vec[63],param_vec[44])*param_vec[107]*y_vec[2])
	buffer[3] = -param_vec[110]*y_vec[3] - (sigmoid(y_vec[56],param_vec[7],param_vec[26],param_vec[63],param_vec[44])*y_vec[7]*(param_vec[109]*y_vec[3]) + (sigmoid(6*G,param_vec[81])*(param_vec[111]*y_vec[10]) + (param_vec[108]*y_vec[2] + sigmoid(y_vec[56],param_vec[8],param_vec[27],param_vec[64],param_vec[45])*(param_vec[107]*y_vec[2]))))
	buffer[4] = -sigmoid(y_vec[56],param_vec[4],param_vec[23],param_vec[60],param_vec[41])*param_vec[92]*y_vec[4] - (sigmoid(6*G,param_vec[79])*param_vec[91]*y_vec[4] + (param_vec[103]*y_vec[1] + (param_vec[110]*y_vec[3] + param_vec[99]*y_vec[10])))
	buffer[5] = -y_vec[9]*param_vec[94]*y_vec[5] - (param_vec[93]*y_vec[5] + (param_vec[102]*y_vec[1] + sigmoid(y_vec[56],param_vec[4],param_vec[23],param_vec[60],param_vec[41])*(param_vec[91]*y_vec[4])))
	buffer[6] = -y_vec[11]*param_vec[97]*y_vec[6] + (param_vec[93]*y_vec[5] + param_vec[96]*y_vec[8])
	buffer[7] = -sigmoid(y_vec[56],param_vec[5],param_vec[24],param_vec[61],param_vec[42])*y_vec[7]*(param_vec[109]*y_vec[3]) + (sigmoid(y_vec[56],param_vec[8],param_vec[27],param_vec[64],param_vec[45])*(param_vec[104]*y_vec[46]) + y_vec[11]*(param_vec[97]*y_vec[6]))
	buffer[8] = -param_vec[96]*y_vec[8] - (param_vec[95]*y_vec[8] + y_vec[9]*(param_vec[94]*y_vec[5]))
	buffer[9] = -y_vec[9]*param_vec[94]*y_vec[5] + (param_vec[100]*y_vec[1] + (param_vec[95]*y_vec[8] + param_vec[98]*y_vec[10]))
	buffer[10] = -param_vec[99]*y_vec[10] - (param_vec[98]*y_vec[10] - (sigmoid(6*G,param_vec[79])*param_vec[111]*y_vec[10] + (param_vec[96]*y_vec[8] + sigmoid(6*G,param_vec[81])*(param_vec[92]*y_vec[4]))))
	buffer[11] = -y_vec[11]*param_vec[97]*y_vec[6] + (param_vec[108]*y_vec[2] + param_vec[99]*y_vec[10])
	buffer[12] = sigmoid(y_vec[56],param_vec[9],param_vec[28],param_vec[65],param_vec[46])*(param_vec[115]*y_vec[13]) - 1*sigmoid(6*G,param_vec[82])*(param_vec[116]*y_vec[12])
	buffer[13] = -param_vec[119]*y_vec[13] - (sigmoid(y_vec[56],param_vec[14],param_vec[33],param_vec[70],param_vec[51])*param_vec[117]*y_vec[13] - (sigmoid(6*G,param_vec[82])*param_vec[115]*y_vec[13] - (sigmoid(6*G,param_vec[85])*param_vec[136]*y_vec[13] + (param_vec[118]*y_vec[14] + (sigmoid(y_vec[56],param_vec[9],param_vec[28],param_vec[65],param_vec[46])*(param_vec[138]*y_vec[41]) + 1*(sigmoid(y_vec[56],param_vec[10],param_vec[29],param_vec[66],param_vec[47])*(param_vec[116]*y_vec[12])))))))
	buffer[14] = -sigmoid(y_vec[56],param_vec[10],param_vec[29],param_vec[66],param_vec[47])*y_vec[18]*(param_vec[137]*y_vec[14]) - (param_vec[121]*y_vec[14] - (param_vec[118]*y_vec[14] + (sigmoid(6*G,param_vec[84])*(param_vec[131]*y_vec[22]) + (param_vec[119]*y_vec[13] + (param_vec[139]*y_vec[45] + sigmoid(y_vec[56],param_vec[13],param_vec[32],param_vec[69],param_vec[50])*(param_vec[117]*y_vec[13]))))))
	buffer[15] = -sigmoid(6*G,param_vec[83])*param_vec[122]*y_vec[15] - (sigmoid(y_vec[56],param_vec[11],param_vec[30],param_vec[67],param_vec[48])*param_vec[120]*y_vec[15] + (param_vec[133]*y_vec[21] + (param_vec[130]*y_vec[22] + (param_vec[121]*y_vec[14] + param_vec[140]*y_vec[47]))))
	buffer[16] = -y_vec[22]*param_vec[126]*y_vec[16] - (param_vec[123]*y_vec[16] + (param_vec[134]*y_vec[21] + sigmoid(y_vec[56],param_vec[11],param_vec[30],param_vec[67],param_vec[48])*(param_vec[122]*y_vec[15])))
	buffer[17] = -sigmoid(y_vec[56],param_vec[12],param_vec[31],param_vec[68],param_vec[49])*param_vec[135]*y_vec[17] - (y_vec[25]*param_vec[124]*y_vec[17] + (param_vec[123]*y_vec[16] + param_vec[128]*y_vec[24]))
	buffer[18] = -sigmoid(y_vec[56],param_vec[16],param_vec[35],param_vec[72],param_vec[53])*y_vec[18]*(param_vec[137]*y_vec[14]) + (sigmoid(y_vec[56],param_vec[13],param_vec[32],param_vec[69],param_vec[50])*(param_vec[143]*y_vec[46]) + y_vec[25]*(param_vec[124]*y_vec[17]))
	buffer[19] = -param_vec[144]*y_vec[19] + sigmoid(y_vec[56],param_vec[13],param_vec[32],param_vec[69],param_vec[50])*(y_vec[18]*(param_vec[137]*y_vec[14]))
	buffer[20] = -param_vec[142]*y_vec[20] + (sigmoid(y_vec[56],param_vec[12],param_vec[31],param_vec[68],param_vec[49])*(param_vec[135]*y_vec[17]) - sigmoid(y_vec[56],param_vec[15],param_vec[34],param_vec[71],param_vec[52])*param_vec[141]*y_vec[20])
	buffer[21] = -param_vec[134]*y_vec[21] - (param_vec[133]*y_vec[21] + (param_vec[125]*y_vec[42] - param_vec[132]*y_vec[21]))
	buffer[22] = -sigmoid(6*G,param_vec[83])*param_vec[131]*y_vec[22] - (param_vec[130]*y_vec[22] - (param_vec[129]*y_vec[22] + (param_vec[128]*y_vec[24] + sigmoid(6*G,param_vec[84])*(param_vec[120]*y_vec[15]))))
	buffer[23] = -y_vec[22]*param_vec[126]*y_vec[16] + (param_vec[132]*y_vec[21] + (param_vec[127]*y_vec[24] + param_vec[129]*y_vec[22]))
	buffer[24] = -param_vec[128]*y_vec[24] - (param_vec[127]*y_vec[24] + y_vec[22]*(param_vec[126]*y_vec[16]))
	buffer[25] = -y_vec[25]*param_vec[124]*y_vec[17] + (param_vec[119]*y_vec[13] + param_vec[130]*y_vec[22])
	buffer[26] = sigmoid(y_vec[56],param_vec[20],param_vec[39],param_vec[76],param_vec[57])*(param_vec[167]*y_vec[27]) - 1*sigmoid(6*G,param_vec[86])*(param_vec[168]*y_vec[26])
	buffer[27] = -sigmoid(y_vec[56],param_vec[18],param_vec[37],param_vec[74],param_vec[55])*param_vec[169]*y_vec[27] - (sigmoid(6*G,param_vec[86])*param_vec[167]*y_vec[27] + (sigmoid(y_vec[56],param_vec[20],param_vec[39],param_vec[76],param_vec[57])*(param_vec[160]*y_vec[41]) + 1*(sigmoid(y_vec[56],param_vec[21],param_vec[40],param_vec[77],param_vec[58])*(param_vec[168]*y_vec[26]))))
	buffer[28] = -param_vec[170]*y_vec[28] + sigmoid(y_vec[56],param_vec[21],param_vec[40],param_vec[77],param_vec[58])*(param_vec[169]*y_vec[27])
	buffer[29] = -y_vec[35]*param_vec[163]*y_vec[29] - (sigmoid(y_vec[56],param_vec[17],param_vec[36],param_vec[73],param_vec[54])*param_vec[150]*y_vec[29] + (param_vec[157]*y_vec[32] + param_vec[170]*y_vec[28]))
	buffer[30] = -y_vec[38]*param_vec[152]*y_vec[30] + (param_vec[158]*y_vec[32] + (param_vec[151]*y_vec[31] + sigmoid(y_vec[56],param_vec[17],param_vec[36],param_vec[73],param_vec[54])*(param_vec[150]*y_vec[29])))
	buffer[31] = -param_vec[151]*y_vec[31] + sigmoid(y_vec[56],param_vec[19],param_vec[38],param_vec[75],param_vec[56])*(param_vec[164]*y_vec[46])
	buffer[32] = -param_vec[159]*y_vec[32] - (param_vec[158]*y_vec[32] - (param_vec[157]*y_vec[32] - (param_vec[155]*y_vec[32] - (param_vec[166]*y_vec[32] + (param_vec[156]*y_vec[39] + param_vec[165]*y_vec[42])))))
	buffer[33] = -param_vec[153]*y_vec[33] + y_vec[38]*(param_vec[152]*y_vec[30])
	buffer[34] = -y_vec[38]*param_vec[152]*y_vec[30] + (param_vec[153]*y_vec[33] + param_vec[159]*y_vec[32])
	buffer[35] = -param_vec[162]*y_vec[35] - (param_vec[161]*y_vec[35] + (param_vec[170]*y_vec[28] - y_vec[35]*param_vec[163]*y_vec[29]))
	buffer[36] = -param_vec[154]*y_vec[36] + (param_vec[162]*y_vec[35] + (param_vec[153]*y_vec[33] + param_vec[161]*y_vec[35]))
	buffer[37] = param_vec[153]*y_vec[33] + sigmoid(y_vec[56],param_vec[17],param_vec[36],param_vec[73],param_vec[54])*(param_vec[150]*y_vec[29])
	buffer[38] = -y_vec[38]*param_vec[152]*y_vec[30] + (param_vec[153]*y_vec[33] + param_vec[159]*y_vec[32])
	buffer[39] = param_vec[155]*y_vec[32] - param_vec[156]*y_vec[39]
	buffer[40] = param_vec[154]*y_vec[36]
	buffer[41] = -param_vec[146]*y_vec[41] - (sigmoid(6*G,param_vec[85])*param_vec[106]*y_vec[41] - (sigmoid(y_vec[56],param_vec[14],param_vec[33],param_vec[70],param_vec[51])*param_vec[160]*y_vec[41] - (sigmoid(y_vec[56],param_vec[18],param_vec[37],param_vec[74],param_vec[55])*param_vec[138]*y_vec[41] + (sigmoid(y_vec[56],param_vec[6],param_vec[25],param_vec[62],param_vec[43])*(param_vec[136]*y_vec[13]) + y_vec[51]))))
	buffer[42] = -param_vec[165]*y_vec[42] - (param_vec[149]*y_vec[42] - (param_vec[125]*y_vec[42] - (param_vec[101]*y_vec[42] + (param_vec[166]*y_vec[32] + y_vec[55]))))
	buffer[43] = -param_vec[145]*y_vec[43] + (param_vec[144]*y_vec[19] + y_vec[53])
	buffer[44] = sigmoid(y_vec[56],param_vec[15],param_vec[34],param_vec[71],param_vec[52])*(param_vec[141]*y_vec[20])
	buffer[45] = -param_vec[139]*y_vec[45] + (param_vec[145]*y_vec[43] + 1*(sigmoid(6*G,param_vec[80])*(param_vec[105]*y_vec[0])))
	buffer[46] = -param_vec[148]*y_vec[46] - (sigmoid(6*G,param_vec[80])*param_vec[164]*y_vec[46] - (sigmoid(y_vec[56],param_vec[5],param_vec[24],param_vec[61],param_vec[42])*param_vec[143]*y_vec[46] - (sigmoid(y_vec[56],param_vec[16],param_vec[35],param_vec[72],param_vec[53])*param_vec[104]*y_vec[46] + (param_vec[145]*y_vec[43] + 1*(sigmoid(y_vec[56],param_vec[19],param_vec[38],param_vec[75],param_vec[56])*(param_vec[105]*y_vec[0]))))))
	buffer[47] = -param_vec[140]*y_vec[47] + y_vec[35]*(param_vec[163]*y_vec[29])
	buffer[48] = param_vec[142]*y_vec[20] - param_vec[147]*y_vec[48]
	buffer[49] = param_vec[149]*y_vec[42] + (param_vec[148]*y_vec[46] + (param_vec[146]*y_vec[41] + param_vec[147]*y_vec[48]))
	buffer[50] = (param_vec[112]*F_carb(t) - y_vec[50])/param_vec[88]
	buffer[51] = (y_vec[50] - y_vec[51])/param_vec[88]
	buffer[52] = (param_vec[113]*F_fat(t) - y_vec[52])/param_vec[89]
	buffer[53] = (y_vec[52] - y_vec[53])/param_vec[89]
	buffer[54] = (param_vec[114]*F_prot(t) - y_vec[54])/param_vec[90]
	buffer[55] = (y_vec[54] - y_vec[55])/param_vec[90]
	buffer[56] = -param_vec[2]*y_vec[56] + (param_vec[59]*y_vec[55] + (param_vec[22]*y_vec[53] + param_vec[3]*y_vec[51]))
	buffer[57] = -param_vec[1]*y_vec[57] + param_vec[78]*(1/y_vec[41])
	buffer[58] = -param_vec[0]*y_vec[58] + 60*param_vec[87]
	return buffer