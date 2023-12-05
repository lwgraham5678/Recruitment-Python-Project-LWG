import Testdefs as td
from TestConfig import P
import scipy as sp
import numpy as np
import math as m
import matplotlib.pyplot as plt

originaldataframe = P.totalmenXmencommreg
row_limit = 300
column_limit = 300
num_generated_sequences = 10
sequence_length = 100
Configuration_itterations = 10
num_networks_per_ds = 10
num_alpha_parameters = 10
num_sequence_per_alpha = 10
data_list = []
dist = sp.stats.poisson
#bounds = {'mu' : (0, 1000)}

dataframe = td.RemoveHighValuedNodes(originaldataframe, row_limit, column_limit) # remove hubs

sequences = td.GetDegreeSequences(dataframe)

alpha_parameter_list = list(np.random.uniform(-1.0,1.0,num_alpha_parameters)) # uniformly sampled covariance parameter
#alpha_parameter_list = [0.5]


row_var = np.var(sequences[0]) 
column_var = np.var(sequences[1])
row_mean = np.average(sequences[0])
column_mean = np.average(sequences[1])
#print('social:')
#print(row_mean, row_var)
#print('sexual:')
#print(column_mean, column_var)


# Ian's transformations
row_tau = m.sqrt(1 + ((row_var-row_mean)/(row_mean**2))) 
column_tau = m.sqrt(1 + ((column_var-column_mean)/(column_mean**2)))

row_theta = m.log(row_mean/row_tau)
column_theta = m.log(column_mean/column_tau)

tau_matrix = np.array([[row_tau, 0],[0, column_tau]])

for alpha in alpha_parameter_list:

    #print(alpha)
    
    Omega = np.array([[1, alpha], [alpha, 1]])

    Sigma = np.matmul(tau_matrix, np.matmul(Omega, tau_matrix))
    
    log_of_means_array = sp.stats.multivariate_normal.rvs(mean = [row_theta, column_theta], cov = Sigma, size = sequence_length)

    #plotting multivariate normal distribution-------------------------------------------
    '''
    x, y = np.mgrid[-2.5:6.5:.01, -2.5:6.5:.01]
    pos = np.dstack((x, y))
    mvdist = sp.stats.multivariate_normal(mean = [row_theta, column_theta], cov = Sigma)
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    plt.xlabel('log of social mean')
    plt.ylabel('log of sexual mean')
    ax2.contourf(x, y, mvdist.pdf(pos))
    plt.show()
    '''

    #log_of_means_array = log_of_means_dist.rvs()
    
    means_array = np.exp(log_of_means_array)
    #print(means_array)

    for i in range(num_sequence_per_alpha):
        row_sequence = []
        column_sequence = []
        
        # splits 2 by 1000 array into two lists
        for mean_pair in means_array:
            row_sequence.append(dist.rvs(mu = mean_pair[0]))
            column_sequence.append(dist.rvs(mu = mean_pair[1])) 
        
        correlation_coeff, p_value = sp.stats.spearmanr(row_sequence, column_sequence)
        #print(correlation_coeff, alpha_parameter_list.index(alpha))
        #print(max(row_sequence), min(row_sequence))
        #print(max(column_sequence), min(column_sequence))
        
        #Degree distribution histogram---------------------------------------------------------------
        '''
        plt.hist([x for x in row_sequence if x < 1000], bins = 100)
        plt.xlabel('Degree')
        plt.ylabel('number of nodes')
        plt.title('Degree distribution of social degree sequence')
        plt.show()
        plt.clf()
        plt.hist([x for x in column_sequence if x < 1000], bins = 100)
        plt.xlabel('Degree')
        plt.ylabel('number of nodes')
        plt.title('Degree distribution of sexual degree sequence')
        plt.show()
        plt.clf()
        '''
        
        overlaps = []
        num_edges_list = []
        overlap_max_degree_list = []

        for n in range(0, num_networks_per_ds):
            row_network = td.itter_ConfigGen_min(row_sequence, Configuration_itterations)
            column_network = td.itter_ConfigGen_min(column_sequence, Configuration_itterations)

            overlap = td.FindOverlapStat(row_network, column_network)  

            row_edges = len(row_network.edges())
            column_edges = len(column_network.edges())

            overlaps.append(overlap)
            num_edges_list.append((row_edges, column_edges))
            overlap_max_degree_list.extend(td.FindOverlapDegreeMax(row_network, column_network))

        data_list.append((correlation_coeff, overlaps, num_edges_list))

print(data_list)
print(overlap_max_degree_list)