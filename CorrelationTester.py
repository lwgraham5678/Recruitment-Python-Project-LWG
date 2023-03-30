import math as m
import scipy as sp
import Testdefs as td
from TestConfig import P
import psython as psy


L = [(P.numpars1Xmencommreg, 'numpars1Xmencommreg:'), (P.numpars1Xnum_bmsm, 'numpars1Xnum_bmsm:'), (P.totalmenXmencommreg, 'totalmenXmencommreg:'), (P.totalmenXnum_bmsm, 'totalmenXnum_bmsm:')]

for f, j in L:
    dsl = td.GetDegreeSequences(f)

    f = []
    g = []

    for i in dsl[0]:
        if i == 0:
            f.append(0)
        else:
            f.append(m.log(i))

    for i in dsl[1]:
        if i == 0:
            g.append(0)
        else:
            g.append(m.log(i))

    x = td.pearsonr_ci(f, g, 0.05)

    print(j, x)