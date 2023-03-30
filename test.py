import Testdefs as td
from TestConfig import P
import numpy as np
import networkx as nx

L = [(P.numpars1Xmencommreg, 'numpars1Xmencommreg:'), (P.numpars1Xnum_bmsm, 'numpars1Xnum_bmsm:'), (P.totalmenXmencommreg, 'totalmenXmencommreg:'), (P.totalmenXnum_bmsm, 'totalmenXnum_bmsm:')]
'''
td.ExpDistribution_TB(600, 200, .5)

'''
for df, name in L:
    X = td.GetDegreeSequences(df)

    dsc = X[0]
    dsx = X[1]

    #A = td.itter_ConfigGen(dsc, 10)
    #B = td.itter_ConfigGen(dsx, 10)

    print('dsc for ' + name + f'{dsc}')
    print('dsx for ' + name + f'{dsx}')
