import matplotlib.pyplot as plt
import pandas
import numpy as np
import scipy as sp
#Exporting-------------------------------------------------------------------------------------------------------------------------------------------------

points = pandas.read_csv(r'C:\Users\lwgra\OneDrive\Documents\School\Fall 2022\UG research\networks and data science\totalmenXnum_bmsm.csv', index_col=0)

X = [int(i) for i in points.columns.to_list()]
Ys = [int(i) for i in points.index]

HMArraystart = points.to_numpy()

Xlabel = 'totalmen'
Ylabel = 'num_bmsm'

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
#plotting 2D-----------------------------------------------------------------------------------------------------------------------------------------------
#print(points)

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
