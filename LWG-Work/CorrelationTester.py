import math as m
import scipy as sp
import numpy as np
import Testdefs as td
from TestConfig import P
#import psython as psy
from data import Data
import math

'''
L = [(P.numpars1Xmencommreg, 'numpars1Xmencommreg:'), (P.numpars1Xnum_bmsm, 'numpars1Xnum_bmsm:'), (P.totalmenXmencommreg, 'totalmenXmencommreg:'), (P.totalmenXnum_bmsm, 'totalmenXnum_bmsm:')]

for f, j in L:
    dsl = td.GetDegreeSequences(f)

    x = td.spearmanr_ci(dsl[0], dsl[1], 0.05)

    print(j, x)
'''
corr = []
ovlp = []
for correlation, overlap_list, edges in Data:
    corr += [correlation for o in overlap_list]
    ovlp += [o for o in overlap_list]


ovlp2 = []
for o in ovlp:
    if o != 0:
        ovlp2.append(math.log(o))
    else:
        ovlp.append(0)

print(sp.stats.pearsonr(corr, ovlp2))