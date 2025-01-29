import numpy as np
import pandas as pd
import matplotlib
import os

from matplotlib import pyplot as plt
from utility import load_data, get_well_data
from models import Filter_type, get_filter, apply_filter, get_MP, print_multiple, get_KDP

# Source file of data
source_file = 'EPIC/Dados/merged_data.csv'

# Load well data
data_full = load_data(source_file)
well_data = get_well_data(data_full, 'A-06 2', only_numerical=True, remove_ints=True, drop_na=True, replace_na=0.)

# Define and apply filter
sampling = 1 # 1 measurement / hour
T_cutoff = np.array([30, 6]) # 6 hour cutoff period
filter_type = Filter_type.BUTTER_PASS
order = 2

cutoff_freq = 1/T_cutoff
num_coef, den_coef = get_filter(filter_type, sampling_freq=sampling, cutoff_freq=cutoff_freq, order=order)
filtered_data = apply_filter(well_data, num_coef, den_coef)

# Calculate Matrix Profile of well data
subseq_size = 24
quartile = 0.75
MP = get_MP(well_data, subseq_size, quartile=quartile, ignore_start=True)

fig, axs = print_multiple(MP, height=1, width=7)


plt.show()