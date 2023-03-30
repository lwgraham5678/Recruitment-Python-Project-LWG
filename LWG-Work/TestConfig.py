import networkx as nx
import pandas
import numpy as np
import scipy as sp
class P:
    g1 = nx.Graph()
    g2 = nx.Graph()

    n = 50 #nodes in graph
    k = 10 #edges in graph

    Cor_1 = [1, 12, 20, 30, 33, 40, 41, 57, 66, 73, 79, 80, 117, 120, 149, 151, 164, 168, 191, 201, 202, 213, 225, 227, 235, 237, 256, 258, 263, 280, 290, 292, 297, 317, 321, 332, 333, 339, 346, 388, 414, 426, 446, 447, 457, 462, 466, 488, 497, 499, 502, 507, 509, 517, 526, 532, 533, 540, 541, 567, 568, 569, 572, 589, 605, 608, 609, 613, 625, 652, 695, 702, 707, 717, 743, 754, 756, 770, 781, 792, 810, 818, 819, 831, 836, 841, 843, 850, 859, 863, 873, 893, 930, 931, 939, 949, 961, 963, 976, 996]
    Cor_2 = [14, 17, 22, 56, 67, 69, 77, 79, 116, 127, 130, 142, 174, 180, 181, 192, 194, 219, 221, 223, 233, 234, 236, 245, 263, 273, 283, 286, 305, 306, 311, 327, 328, 357, 368, 394, 398, 404, 405, 407, 413, 414, 426, 439, 454, 462, 466, 469, 485, 496, 497, 508, 558, 567, 585, 586, 597, 616, 618, 619, 625, 628, 644, 656, 664, 675, 690, 702, 716, 734, 743, 747, 755, 756, 757, 759, 768, 769, 776, 784, 818, 820, 831, 854, 864, 877, 882, 892, 911, 919, 928, 940, 945, 948, 952, 953, 956, 966, 996, 999]

    desr = 0.9957387249440693
    bw = 0.2

    totalmenXmencommreg = pandas.read_csv(r'.\RefrenceFiles\totalmenXmencommreg.csv', index_col=0) #gets data from local spot and reads it into a data frame
    numpars1Xmencommreg = pandas.read_csv(r'.\RefrenceFiles\numpars1Xmencommreg.csv', index_col=0)
    numpars1Xnum_bmsm   = pandas.read_csv(r'.\RefrenceFiles\numpars1Xnum_bmsm.csv', index_col=0)
    totalmenXnum_bmsm   = pandas.read_csv(r'.\RefrenceFiles\totalmenXnum_bmsm.csv', index_col=0)

    L = [(numpars1Xmencommreg, 'numpars1Xmencommreg:'), (numpars1Xnum_bmsm, 'numpars1Xnum_bmsm:'), (totalmenXmencommreg, 'totalmenXmencommreg:'), (totalmenXnum_bmsm, 'totalmenXnum_bmsm:')]

    overlapstat = 0.0
    overlapstatlist = []

