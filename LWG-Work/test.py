import Testdefs as td
from TestConfig import P
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy as sp
import math as m
import json
'''
for i, j in P.L:
    print(j)
    ds = td.RemoveHighValuedNodes(i, 300, 300)
    X = td.GetDegreeSequences(ds)
    a, loc, scale = td.Fit(X[1], 'powerlaw')
    Nds1 = sp.stats.powerlaw.rvs(a, loc, scale, size = 5000)

    Nds2 = [round(i) for i in Nds1]
    print(len(Nds2))
    td.itter_ConfigGen(list(Nds2), 100)


#fd = 
#print(P.L[0][0])
fd = td.GetDegreeDistrabution(td.RemoveHighValuedNodes(P.L[0][0], 10, 10))
td.Fit(fd[0], 'powerlaw')
print(fd)


points = td.RemoveHighValuedNodes(P.numpars1Xnum_bmsm, 300, 300)

X = td.GetDegreeSequences(points)

print(sp.stats.spearmanr(X[0],X[1]), td.spearmanr_ci(X[0],X[1], 0.05))

td.degreedistbar(X[0], title = 'Degree distribution of social degree sequence')
td.degreedistbar(X[1], title = 'Degree distribution of sexual degree sequence')
'''

G1 = nx.configuration_model([1,2,3,4])
G2 = nx.configuration_model([0,1,3,2])

print(G1.edges(), G2.edges())

print(td.FindOverlapDegreeMax(G1, G2))