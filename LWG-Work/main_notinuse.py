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
num_networks_per_ds = 50
data_list = []
num_alpha_parameters = 50

dataframe = td.RemoveHighValuedNodes(originaldataframe, row_limit, column_limit)

sequences = td.GetDegreeSequences(dataframe)

alpha_parameter_list = list(np.random.uniform(-1.0,1.0,num_alpha_parameters))
#row_a, row_loc, row_scale = td.Fit(sequences[0], 'powerlaw')
#column_a, column_loc, column_scale = td.Fit(sequences[1], 'powerlaw')

for i in range(0, num_generated_sequences):
    #row_ds = [round(k) for k in sp.stats.powerlaw.rvs(row_a, row_loc, row_scale, sequence_length)]
    #column_ds = [round(k) for k in sp.stats.powerlaw.rvs(column_a, column_loc, column_scale, sequence_length)]

    corr_coef, p_value = sp.stats.spearmanr(row_ds, column_ds)
    confidence_interval = td.spearmanr_ci(row_ds, column_ds, 0.05)

    overlaps = []

    for n in range(0, num_networks_per_ds):
        row_network = td.itter_ConfigGen_min(row_ds, Configuration_itterations)
        column_network = td.itter_ConfigGen_min(column_ds, Configuration_itterations)

        overlap = td.FindOverlapStat(row_network, column_network)

        overlaps.append(overlap)
        #print(n)

    ds_entry = (corr_coef, confidence_interval, overlaps)

    data_list.append(ds_entry)

for ele in data_list:
    print(ele)