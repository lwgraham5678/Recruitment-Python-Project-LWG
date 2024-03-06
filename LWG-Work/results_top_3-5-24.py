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

print(tau_matrix)
print(row_theta)
print(column_theta)