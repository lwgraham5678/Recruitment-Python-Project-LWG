import Testdefs as td
from TestConfig import P
import scipy as sp
import numpy as np


originaldataframe = P.totalmenXnum_bmsm
row_limit = 100
column_limit = 100
num_generated_sequences = 10
sequence_length = 5000
Configuration_itterations = 100
num_networks_per_ds = 10
num_alpha_parameters = 10
data_list = []

dataframe = td.RemoveHighValuedNodes(originaldataframe, row_limit, column_limit)

sequences = td.GetDegreeSequences(dataframe)

alpha_parameter_list = list(np.random.uniform(-1.0,1.0,num_alpha_parameters))

for alpha in alpha_parameter_list:

    Omega = np.array([[1, alpha], [alpha, 1]])
    row_sigma = sp.stats.poisson.std(sequences[0])
    column_sigma = sp.stats.piosson.std(sequences[1])

    Sigma = np.matmul(np.array([[row_sigma,0],[0,column_sigma]]), np.matmul(np.array(Omega, np.array([[row_sigma,0],[0,column_sigma]]))))