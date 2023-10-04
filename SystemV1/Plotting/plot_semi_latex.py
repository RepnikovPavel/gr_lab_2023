import matplotlib.pyplot as plt
import numpy as np

def latex_float(number,precision=2):
    float_str = ''
    if precision == 3:
        float_str="{0:.3g}".format(number)
    if precision == 2:
        float_str="{0:.2g}".format(number)
    if precision == 1:
        float_str="{0:.1g}".format(number)
    if precision == 4:
        float_str="{0:.4g}".format(number)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return '$'+r"{0} \times 10^{{{1}}}".format(base, int(exponent))+'$'
    else:
        return float_str