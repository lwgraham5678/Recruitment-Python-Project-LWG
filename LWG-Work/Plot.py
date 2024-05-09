import matplotlib.pyplot as plt
import pandas
import numpy as np
import scipy as sp
from TestConfig import P
import Testdefs as td
#Exporting-------------------------------------------------------------------------------------------------------------------------------------------------
'''
points = P.numpars1Xnum_bmsm

X = [int(i) for i in points.columns.to_list()]
Ys = [int(i) for i in points.index]

HMArraystart = points.to_numpy()

Xlabel = 'numpars1'
Ylabel = 'num_bmsm'
'''
#plotting 3D-----------------------------------------------------------------------------------------------------------------------------------------------
'''
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

for i in range(0, len(Ys)):
    Z = [int(j) for j in points.iloc[i]]
    Y = [Ys[i] for j in range(0, len(X))]
    ax.scatter(X, Y, Z, marker='o')

ax.set_xlabel(Xlabel)
ax.set_ylabel(Ylabel)
ax.set_zlabel('n')

plt.show()
'''
#plotting Heat---------------------------------------------------------------------------------------------------------------------------------------------
'''
HMArray = np.flip(HMArraystart,axis=0)

fig, ax = plt.subplots()
im = ax.imshow(HMArray)

Y = Ys[::-1]

ax.set_xticks(np.arange(len(X)), labels=X)
ax.set_yticks(np.arange(len(Y)), labels=Y)

xticks = ax.xaxis.get_major_ticks()
yticks = ax.yaxis.get_major_ticks()

#print(X)
#print([xticks[X.index(a)] for a in X])

for ele in xticks:
    if (xticks.index(ele) % 2 == 0):
        xticks[xticks.index(ele)].set_visible(False)

for ele in yticks:
    if(yticks.index(ele) % 2 == 0):
        yticks[yticks.index(ele)].set_visible(False)

ax.set_xlabel(Xlabel)
ax.set_ylabel(Ylabel)

plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
         rotation_mode="anchor")

"""
for i in range(len(X)):
    for j in range(len(Ys)):
        text = ax.text(j, i, HMArray[i, j], ha="center", va="center", color="w")
"""
ax.set_title("Degree Correlation Heat Map")
fig.tight_layout()
plt.show()
'''
#plotting scatter-----------------------------------------------------------------------------------------------------------------------------------------------
#print(points)
'''
Xf =[]
Yf = []

for i in range(len(X)):
    for j in range(len(Ys)):
        ele = HMArraystart[j, i]
        if ele != 0:
            for k in range(0,ele):
                Xf.append(X[i])
                Yf.append(Ys[j])

#for i in range(len(Xf)):
#    print(Xf[i],Yf[i])

print(sp.stats.pearsonr(Xf,Yf))
plt.scatter(Xf,Yf)
plt.xlabel(Xlabel)
plt.ylabel(Ylabel)
plt.show()
'''
# Violin Plot-----------------------------------------------------------------------------------------------------------------------------------------------
'''
Data = P.data_2_6_2024

cor_coef = [a for a, b, c in Data]
#print(cor_coef)
overlap_dist = [b for a, b, c in Data]
zipped_num_edges = [c for a, b, c in Data]

bins = [-0.5*a for a in range(1,20)] + [0.5*a for a in range(1,20)] + [0.0]

#combined_corr = {a : [] for a in bins}
#print(combined_corr.keys())
#for coef, overlap in zip(cor_coef, overlap_dist):
#    for num in combined_corr.keys():
#        if coef < num + 0.025 and coef > num - 0.025:
#            combined_corr[num] = combined_corr[num] + overlap
#print([a for a in combined_corr.keys() if combined_corr[a] != []])

true_set = { x : y for x , y in combined_corr.items() if y != []}

plt.xlabel('Correlation coefficient')
plt.ylabel('Overlap Distribution')
plt.violinplot(dataset = overlap_dist, positions = cor_coef, showmeans = True)
#plt.violinplot(dataset = list(true_set.values()), positions = list(true_set.keys()), showmeans = True)
plt.show()

plt.clf()

Xrow_num_edges = []
YOverlap = []

for i in range(0,len(overlap_dist)):
    Xrow_num_edges += [a for a, b in zipped_num_edges[i]]
    YOverlap += overlap_dist[i]

print('social:')
print(sp.stats.spearmanr(Xrow_num_edges, YOverlap), td.spearmanr_ci(Xrow_num_edges, YOverlap, 0.05))

plt.xlabel('Number of edges in social network')
plt.ylabel('Overlap')
plt.scatter(Xrow_num_edges, YOverlap)
plt.show()

plt.clf()

Xcolumn_num_edges = []
YOverlap = []

for i in range(0,len(overlap_dist)):
    Xcolumn_num_edges += [b for a, b in zipped_num_edges[i]]
    YOverlap += overlap_dist[i]

print('sexual:')
print(sp.stats.spearmanr(Xcolumn_num_edges, YOverlap), td.spearmanr_ci(Xcolumn_num_edges, YOverlap, 0.05))

plt.xlabel('Number of edges in sexual network')
plt.ylabel('Overlap')
plt.scatter(Xcolumn_num_edges, YOverlap)
plt.show()

plt.clf()
'''
#Max Degree Bar chart---------------------------------------------------------------------------------------------

data = P.max_deg_list

max_local_deg = [a for (a, b, c) in data]
gc_deg = [b for (a, b, c) in data]
gx_deg = [c for (a, b, c) in data]

plt.xlabel('Local overlap maximmum')
plt.ylabel('social network degree')
plt.scatter(max_local_deg, gc_deg)
plt.show()


plt.xlabel('Local overlap maximmum')
plt.ylabel('sexual network degree')
plt.scatter(max_local_deg, gx_deg)
plt.show()