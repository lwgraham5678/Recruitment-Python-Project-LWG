import Testdefs as td
from TestConfig import P
import scipy as sp
import numpy as np


originaldataframe = P.totalmenXnum_bmsm
row_limit = 300
column_limit = 300
num_generated_sequences = 10
sequence_length = 1000
Configuration_itterations = 100
num_networks_per_ds = 10
num_alpha_parameters = 10
data_list = []
dist = sp.stats.poisson
bounds = {'mu' : (0, 200)}

dataframe = td.RemoveHighValuedNodes(originaldataframe, row_limit, column_limit)

sequences = td.GetDegreeSequences(dataframe)

alpha_parameter_list = list(np.random.uniform(-1.0,1.0,num_alpha_parameters))

row_fit = sp.stats.fit(dist, sequences[0], bounds)
column_fit = sp.stats.fit(dist, sequences[1], bounds)

row_mu, row_loc = row_fit.params
column_mu, column_loc = column_fit.params

row_std = dist.std(row_mu)
column_std = dist.std(column_mu)
row_mean = dist.mean(row_mu)
column_mean = dist.mean(column_mu)


std_matrix = np.array([[row_mean, 0],[0, column_mean]])

for alpha in alpha_parameter_list:

    print(alpha)
    
    Omega = np.array([[1, alpha], [alpha, 1]])

    Sigma = np.matmul(std_matrix, np.matmul(Omega, std_matrix))
    
    log_of_means_dist = sp.stats.multivariate_normal([row_mean, column_mean], Sigma)

    log_of_means_array = log_of_means_dist.rvs()
    
    means_array = np.exp(log_of_means_array)

    row_sequence = list(dist.rvs(mu = means_array[0], loc = row_loc, size = sequence_length))
    column_sequence = list(dist.rvs(mu = means_array[1], loc = column_loc, size = sequence_length))
    correlation_coeff, p_value = sp.stats.spearmanr(row_sequence, column_sequence)
    print(row_sequence, column_sequence)
    
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