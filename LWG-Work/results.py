import Testdefs as td
from TestConfig import P
import scipy as sp
import numpy as np
import math as m
import matplotlib.pyplot as plt

originaldataframe = P.numpars1Xnum_bmsm
row_limit = 100
column_limit = 100
num_generated_sequences = 10
sequence_length = 1000
Configuration_itterations = 10
num_networks_per_ds = 50
num_alpha_parameters = 50
data_list = []
dist = sp.stats.poisson
bounds = {'mu' : (0, 1000)}

dataframe = td.RemoveHighValuedNodes(originaldataframe, row_limit, column_limit)

sequences = td.GetDegreeSequences(dataframe)

alpha_parameter_list = list(np.random.uniform(-1.0,1.0,num_alpha_parameters))

row_fit = sp.stats.fit(dist, sequences[0], bounds)
column_fit = sp.stats.fit(dist, sequences[1], bounds)

row_mu, row_loc = row_fit.params
column_mu, column_loc = column_fit.params

row_std = dist.std(row_mu, row_loc)
column_std = dist.std(column_mu, column_loc)
row_mean = row_mu
column_mean = column_mu


std_matrix = np.array([[m.log(row_std), 0],[0, m.log(column_std)]])

for alpha in alpha_parameter_list:

    #print(alpha)
    
    Omega = np.array([[1, alpha], [alpha, 1]])

    #Sigma = np.array([[(row_std**2)*alpha, row_std*column_std], [row_std*column_std, (column_std**2)*alpha]]) 
    Sigma = np.matmul(std_matrix, np.matmul(Omega, std_matrix))
    
    log_of_means_array = sp.stats.multivariate_normal.rvs(mean = [m.log(row_mean), m.log(column_mean)], cov = Sigma)

    #plotting sub routine-------------------------------------------
    '''
    x, y = np.mgrid[0.0:3.5:.01, 0.0:3.5:.01]
    pos = np.dstack((x, y))
    mvdist = sp.stats.multivariate_normal(mean = [m.log(row_mean), m.log(column_mean)], cov = Sigma)
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.contourf(x, y, mvdist.pdf(pos))
    plt.show()
    '''

    #log_of_means_array = log_of_means_dist.rvs()
    
    means_array = np.exp(log_of_means_array)
    #print(means_array)

    row_sequence = list(dist.rvs(mu = means_array[0], size = sequence_length))
    column_sequence = list(dist.rvs(mu = means_array[1], size = sequence_length))
    correlation_coeff, p_value = sp.stats.spearmanr(row_sequence, column_sequence)
    #print(correlation_coeff)
    #print(max(row_sequence), min(row_sequence))
    #print(max(column_sequence), min(column_sequence))
    
    overlaps = []
    num_edges_list = []

    for n in range(0, num_networks_per_ds):
        row_network = td.itter_ConfigGen_min(row_sequence, Configuration_itterations)
        column_network = td.itter_ConfigGen_min(column_sequence, Configuration_itterations)

        overlap = td.FindOverlapStat(row_network, column_network)  

        row_edges = len(row_network.edges())
        column_edges = len(column_network.edges())

        overlaps.append(overlap)
        num_edges_list.append((row_edges, column_edges))

    data_list.append((correlation_coeff, overlaps, num_edges_list))

print(data_list)