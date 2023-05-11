import networkx as nx
import scipy as sp
import numpy as np
from collections import OrderedDict
import pandas
import math as m
import playsound as snd
import matplotlib.pyplot as plt

def AddRandomEdge(g: nx.Graph, n: int):
    #this function adds a random edge to a graph
    
    P = False
    while not P:
        x = np.random.randint(0,n)
        y = np.random.randint(0,n)
        if x != y:
            t = (x,y)
            P = True
    
    g.add_edge(*t)

def NetworkCorrolation(g1: nx.Graph, g2: nx.Graph):
    #this function calculates correlation between edges in two networks (not in use)

    dict1 = {i:j for (i,j) in g1.degree()}
    dict2 = {i:j for (i,j) in g2.degree()}
    kl1 = dict1.keys()
    kl2 = dict2.keys()

    ml = [max(kl1),max(kl2)]

    for k in range(0,max(ml)):
        if not k in kl1:
            dict1[k] = 0
        if not k in kl2:
            dict2[k] = 0
    

    dict1 = OrderedDict(sorted(dict1.items()))
    dict2 = OrderedDict(sorted(dict2.items()))

    rt = sp.stats.pearsonr(list(dict1.values()),list(dict2.values()))
    
    return rt

def PermuteSequence(S1: list,S2: list,desr: float,bw: float):
    #this function permutes a list until it's correlation with a refrence list is in a badwidth of a desired r value.
    
    p = False
    tc = 0
    PA = 0
    while not p:
        rt = sp.stats.pearsonr(S1,S2)
        r = rt.statistic
        tc += 1

        if desr-(bw/2)<r<desr+(bw/2):
            p = True
            
            return [S1, S2, rt]#make into class
        elif tc > 999:
            print(PA/tc)
            raise TimeoutError("Skill issue: correlation failed to enter bandwidth in specified computing time")
        else:
            p = False
        
        q = False

        while not q:
            x = np.random.randint(0,len(S1))
            y = np.random.randint(0,len(S1))

            if x != y:
                q = True
        
        NS1 = S1
        Ay = NS1[y]
        Ax = NS1[x]

        NS1[y] = Ax
        NS1[x] = Ay

        nrt = sp.stats.pearsonr(NS1,S2)
        nr = nrt.statistic
        
        if nr>r:
            S1 = NS1#pass criteria may need to add p-value criteria ask bree
            PA += 1

def ScrableSequence(L1: list, Perm: int):
    #this function swaps random elements in a list Perm times
    
    for i in range(0, Perm):
        x = np.random.randint(0,len(L1))
        y = np.random.randint(0,len(L1))

        rp1 = L1[x]
        rp2 = L1[y]

        L1[y] = rp1
        L1[x] = rp2

    return L1
    
def rotate(l, n):
    return l[n:] + l[:n]

def GetDegreeSequences(Df: pandas.DataFrame):
    #function extracts two degree sequence lists from a cross tab dataframe
    
    HMArraystart = Df.to_numpy()

    X = [int(i) for i in Df.columns.to_list()]
    Ys = [int(i) for i in Df.index]

    Xf = []
    Yf = []

    for i in range(len(X)):
        for j in range(len(Ys)):
            ele = HMArraystart[j, i]
            if ele != 0:
                for k in range(0,ele):
                    Xf.append(X[i])
                    Yf.append(Ys[j])
    
    return([Xf, Yf])

def GetDegreeDistrabution(Df: pandas.DataFrame):
    #function extracts two degree distrabutions lists from a cross tab dataframe
    
    HMArraystart = Df.to_numpy()

    X = [int(i) for i in Df.columns.to_list()]
    Y = [int(i) for i in Df.index]

    Xf = {i : 0 for i in X}
    Yf = {i : 0 for i in Y}

    for i in range(len(X)):
        for j in range(len(Y)):
            ele = HMArraystart[j, i]
            Xf[X[i]] += ele
            Yf[Y[j]] += ele
    return([Xf, Yf])

def CreateNetworks(dsc: list, dsx: list):
    #function configures social and sexual networks then extracts overlap statistic from them. (split into two functions)

    gc = nx.configuration_model(dsc, create_using=nx.Graph)
    gx = nx.configuration_model(dsx, create_using=nx.Graph)

    return [gc, gx]


def FindOverlapStat(gc : nx.graph, gx : nx.graph):
    #Given two networks this function finds the overlap between the two ie (# of shared edges)/(total # of edges in gc)
    
    elc = gc.edges()
    elx = gx.edges()

    Olp = len(set(elc).intersection(set(elx)))

    nc = len(gc.edges())

    stat = Olp / nc

    return(stat)

def oddtest(l : list):
     #Given a list this function test if the sum is odd. If so it adds one to the max entry and returns the new list. If not it returns the list as is.

    if not((sum(l) % 2) == 0):
        x = max(l)
        xi = l.index(x)
        l[xi] = x + 1
        return l
    else:
        return l
        
def StarterGraph(n : int, o : float):
    #Given a number of nodes (n) and an ovelap (o) this funtion generates a graph with n nodes and o times n unique* edges

    g = nx.Graph()
    g.add_nodes_from(0,n)
    e = m.floor(n*o)

    for i in range(1, e):
        x = True
        while x:
            u = np.random.randint(0,n)
            v = np.random.randint(0,n)
            if u != v:
                g.add_edge(u,v) #need a uniqueness condidtion.
                x = False
    return g

def cronbach_alpha(df):
    #I did not write this function
    
    # 1. Transform the df into a correlation matrix
    df_corr = df.corr()
    
    # 2.1 Calculate N
    # The number of variables equals the number of columns in the df
    N = df.shape[1]
    
    # 2.2 Calculate R
    # For this, we'll loop through the columns and append every
    # relevant correlation to an array calles "r_s". Then, we'll
    # calculate the mean of "r_s"
    rs = np.array([])
    for i, col in enumerate(df_corr.columns):
        sum_ = df_corr[col][i+1:].values
        rs = np.append(sum_, rs)
    mean_r = np.mean(rs)
    
   # 3. Use the formula to calculate Cronbach's Alpha 
    cronbach_alpha = (N * mean_r) / (1 + (N - 1) * mean_r)
    return cronbach_alpha

def pearsonr_ci(x, y, alpha):
    #calculates confidence intervals based on pearson r 
    r, p = sp.stats.pearsonr(x, y)

    r_z = np.arctanh(r)
    se = 1/np.sqrt(len(x)-3)
    z = sp.stats.norm.ppf(1-alpha/2)
    lo_z = r_z-z*se
    hi_z = r_z+z*se
    lo = np.tanh(lo_z)
    hi = np.tanh(hi_z)

    return (lo, hi)

def spearmanr_ci(x, y, alpha):
    #calculates confidence intervals based on spearman r

    r, p = sp.stats.spearmanr(x, y)

    r_z = np.arctanh(r)
    se = 1/np.sqrt(len(x)-3)
    z = sp.stats.norm.ppf(1-alpha/2)
    lo_z = r_z-z*se
    hi_z = r_z+z*se
    lo = np.tanh(lo_z)
    hi = np.tanh(hi_z)

    return (lo, hi)

def ConfigGen(ds : list, extra_values = False):
    #Generates a simple graph using configuration model, optional return data from the graph
    
    ds = oddtest(ds)
    G = nx.configuration_model(ds)
    M = len(G.edges())
    G.remove_edges_from(nx.selfloop_edges(G))
    G = nx.Graph(G) # cast to Graph to remove multi edges
    #nx.draw_random(G.subgraph(list(G.nodes)[0:60]), node_size = 0.85)
    #plt.show()
    if extra_values:
        n = (len(G.edges()))
        Z = M - n
        return [G, Z, M]
    else:
        return G

def itter_ConfigGen(ds: list, num_itters : int):
    #itterates ConfigGen and checks to see if simple graph matches degree sequence

    L = []
    
    X = ConfigGen(ds, extra_values = True)
    G = X[0]
    n = X[1] # Z = M - n
    N = X[2] # M
    c = 0

    while (n/N) > 0 and c < num_itters:
        X = ConfigGen(ds, extra_values = True)
        G = X[0]
        n = X[1]
        N = X[2]
        c += 1
        L.append(n/N)

    if (n/N) == 0:
        print("success")
        return G

    while (n/N) > 0.05 and c < num_itters:
        X = ConfigGen(ds, extra_values = True)
        G = X[0]
        n = X[1]
        N = X[2]
        c += 1
        L.append(n/N)

    if (n/N) < 0.05:
        print(min(L), max(L))
        return G
    else:
        pipe()
        print(min(L), max(L))
        #raise TimeoutError("Skill Issue: itterations failed to bring graph into band width")
        return [G, min(L)]
    
def itter_ConfigGen_min(ds : list, num_itters : int):
    # function that takes a degree sequence runs it through the config model multiple times and returns the one with the lowest n/N
    network_dictionary = {}

    for i in range(0, num_itters):
        X = ConfigGen(ds, extra_values = True)
        G = X[0]
        n = X[1] # Z = M - n
        N = X[2] # M

        network_dictionary[n/N] = G
    
    best_network = min(network_dictionary.keys())
    #print([network_dictionary[best_network].edges(), network_dictionary[best_network].nodes()])
    return network_dictionary[best_network]


def PowerDistribution(n : int, max : int, alpha : float):
    Li = sp.stats.powerlaw.rvs(alpha, loc = 0, scale = max, size = n)
    L = [round(x) for x in Li]
    return L

def count_occurrence(lst, x):
   count = 0
   for item in lst:
      if (item == x):
         count = count + 1
   return count

def ExpDistribution(n : int, scalar : int, beta : float):
    Dis = np.random.exponential(beta, n)
    NDis = [round(scalar*x) for x in Dis]
    return NDis

def degreedistbar(ds, title = ''):
    
    Ld = {}

    for i in set(ds):

        Ld[i] = count_occurrence(ds, i)

        if type(i) != int or i < 0:
            pipe()
            raise TypeError('Skill Issue: List entry not positive integer')
    print(max(Ld.values()))
    plt.bar(list(Ld.keys()), list(Ld.values()), width= 0.5)
    plt.xlabel('Degree')
    plt.ylabel('number of nodes')
    plt.title(title)
    plt.show()


def RemoveHighValuedNodes(df : pandas.DataFrame, Xlimit : int, Ylimit : int):
    # removes values of a data frame larger than a specified index
    collist = []
    rowlist = []
    
    for col in df.columns:
        if int(col) > Ylimit:
            df[col] = [0 for i in range(0, len(df[col]))]

    for row in df.index:
        if int(row) > Xlimit:
            df.loc[row] = [0 for i in range(0, len(df.loc[row]))]
            
    
    
    return df

def Fit(ds : list, distribution : str):
    # General fontinuous fit function
    dist = getattr(sp.stats, distribution)
    param = dist.fit(ds)
    return param

def RandomValues(parameters, distribution : str, length : int):
    parameters.append(length)
    dist = getattr(sp.stats, distribution)
    values = dist.rvs(tuple(parameters))
    return values

def NormalizedZeroPaddedDS(DD : dict):

    DS = []

    for key, value in DD:
        for i in range(1, value):
            DS.append(key)

    pop = len(DS)
    
    for i in DS:
        i = i/pop

    return DS
# Test benches below

def ConfigGen_TB(n : int, m : int):
    #test bench for Config gen function

    tds = [np.random.randint(0,n) for i in range(0,m)]

    G = ConfigGen(tds)

    for (i,j) in G.edges():
        if i == j:
            print('self edge fail')
            break
    
        G.remove_edge(i,j)

        if (i,j) in G.edges():
            print('duplicate edge fail')
            break

    print('complete')
    
def itter_ConfigGen_TB(n : int, m : int, num_itters : int):
    
    tds = PowerDistribution(n, m, 0.4)

    G = itter_ConfigGen(tds, num_itters)

    tds = oddtest(tds)

    N = sum(tds)
    n = len(G.edges())

    print(f'for this graph n/N = {n/N}')
    print('complete')

def PowerDistribution_TB(n : int, m : int, alpha : float):

    L = PowerDistribution(n, m, alpha)

    Lp = {}

    for i in set(L):
        
        Lp[i] = count_occurrence(L, i)
        
        if type(i) != int or i < 0:
            pipe()
            raise TypeError("Skill Issue: List entry not positive integer")

    plt.bar(Lp.keys(), Lp.values())

    plt.show()

def ExpDistribution_TB(n : int, scalar : int, beta : float):

    L = ExpDistribution(n, scalar, beta)

    Ld = {}

    for i in set(L):

        Ld[i] = count_occurrence(L, i)

        if type(i) != int or i < 0:
            pipe()
            raise TypeError('Skill Issue: List entry not positive integer')
        
    plt.bar(Ld.keys(), Ld.values())
    plt.xlabel('Degree')
    plt.ylabel('number of nodes')
    plt.show()

#PIPE

def pipe():
    #notifaction sound function
    
    snd.playsound(r'.\LWG-Work\ReferenceFiles\PipeFalling.mp3')