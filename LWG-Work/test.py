import Testdefs as td
from TestConfig import P
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy as sp
import math as m

for i, j in P.L:
    print(j)
    ds = td.RemoveHighValuedNodes(i, 300, 300)
    X = td.GetDegreeSequences(ds)
    a, loc, scale = td.Fit(X[1], 'powerlaw')
    Nds1 = sp.stats.powerlaw.rvs(a, loc, scale, size = 5000)

    Nds2 = [round(i) for i in Nds1]
    td.itter_ConfigGen(list(Nds2), 100)

'''
#fd = 
#print(P.L[0][0])
fd = td.GetDegreeDistrabution(td.RemoveHighValuedNodes(P.L[0][0], 10, 10))
td.Fit(fd[0], 'powerlaw')
print(fd)
'''