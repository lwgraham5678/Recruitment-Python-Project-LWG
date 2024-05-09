import Testdefs as td
from TestConfig import P
import scipy as sp
import numpy as np
import math as m
import matplotlib.pyplot as plt
from numba import njit

originaldataframe = P.totalmenXmencommreg

def results(input : int, originaldataframe) :

    row_limit = 300
    column_limit = 300
    Configuration_itterations = 10
    sequence_length = 100
    dist = sp.stats.poisson
    data_list = []

    dataframe = td.RemoveHighValuedNodes(originaldataframe, row_limit, column_limit) # remove hubs
    sequences = td.GetDegreeSequences(dataframe)
    
    alpha_parameter_list = list(np.random.uniform(-1.0,1.0,1))
    alpha = alpha_parameter_list[0]

    row_var = np.var(sequences[0]) 
    column_var = np.var(sequences[1])
    row_mean = np.average(sequences[0])
    column_mean = np.average(sequences[1])
    
    # Ian's transformations
    row_tau = m.sqrt(1 + ((row_var-row_mean)/(row_mean**2))) 
    column_tau = m.sqrt(1 + ((column_var-column_mean)/(column_mean**2)))

    row_theta = m.log(row_mean/row_tau)
    column_theta = m.log(column_mean/column_tau)

    tau_matrix = np.array([[row_tau, 0],[0, column_tau]])

    Omega = np.array([[1, alpha], [alpha, 1]])

    Sigma = np.matmul(tau_matrix, np.matmul(Omega, tau_matrix))
    
    log_of_means_array = sp.stats.multivariate_normal.rvs(mean = [row_theta, column_theta], cov = Sigma, size = sequence_length)

    means_array = np.exp(log_of_means_array)

    row_sequence = []
    column_sequence = []
        
    # splits 2 by 1000 array into two lists
    for mean_pair in means_array:
        row_sequence.append(dist.rvs(mu = mean_pair[0]))
        column_sequence.append(dist.rvs(mu = mean_pair[1])) 
        
    correlation_coeff, p_value = sp.stats.spearmanr(row_sequence, column_sequence)

    overlaps = []
    num_edges_list = []

    row_network = td.itter_ConfigGen_min(row_sequence, Configuration_itterations)
    column_network = td.itter_ConfigGen_min(column_sequence, Configuration_itterations)

    overlap = td.FindOverlapStat(row_network, column_network)  

    row_edges = len(row_network.edges())
    column_edges = len(column_network.edges())

    overlaps.append(overlap)
    num_edges_list.append((row_edges, column_edges))
    #overlap_max_degree_list.append(td.FindOverlapDegreeMax(row_network, column_network))

    data_list.append((correlation_coeff, overlaps, num_edges_list))

    return data_list
