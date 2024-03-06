import Testdefs as td
from TestConfig import P
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy as sp
import math as m
import json


def update_G_dict(G, G_dict):
    
    fail_count = 0
    
    for G_p in G_dict.keys():
        if nx.utils.graphs_equal(G, G_p):
            G_dict[G_p] += 1
        else:
            fail_count += 1

    if fail_count == len(G_dict.keys()):
        G_dict[G] = 1

    return G_dict



ds = [3,3,4,1,2,1]
'''
G1 = nx.empty_graph(5)
G1.add_edges_from([(4,0), (0,4), (3,2), (2,1), (1,0)])
G2 = nx.empty_graph(5)
G2.add_edges_from([(0,1), (1,2), (2,0), (0,3), (3,4)])
G3 = nx.empty_graph(5)
G2.add_edges_from([(0,3), (2,3), (2,0), (0,1), (1,4)])
G4 = nx.empty_graph(5)
G4.add_edges_from([(0,3), (3,1), (1,0), (0,2), (2,4)])

G1_count = 0
G2_count = 0
G3_count = 0
G4_count = 0
'''
G_dict = {}

for i in range(1000):
    G = nx.random_degree_sequence_graph(ds, tries=100)
    
    G_dict = update_G_dict(G, G_dict)


print(len(G_dict.keys()))
print(G_dict.values())

plt.bar(range(len(G_dict.values())), G_dict.values())
plt.show()