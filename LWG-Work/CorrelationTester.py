import math as m
import scipy as sp
import Testdefs as td
from TestConfig import P
import psython as psy


L = [(P.numpars1Xmencommreg, 'numpars1Xmencommreg:'), (P.numpars1Xnum_bmsm, 'numpars1Xnum_bmsm:'), (P.totalmenXmencommreg, 'totalmenXmencommreg:'), (P.totalmenXnum_bmsm, 'totalmenXnum_bmsm:')]

for f, j in L:
    dsl = td.GetDegreeSequences(f)

    x = td.spearmanr_ci(dsl[0], dsl[1], 0.05)

    print(j, x)