import numpy as np
import pandas as pd

from time import perf_counter
from matplotlib import pyplot as plt
from pyfade import load_data, get_well_data, get_sub_sequence
from pyfade import Filter_type, get_filter, apply_filter, get_MP, print_multiple, get_KDP, get_MP_from_wavelets, get_signal_decomp

# Source file of data
source_file = 'EPIC/Dados/merged_data.csv'

# Load well data
data_full = load_data(source_file)

well_data = get_well_data(data_full, 'A-08 2', only_numerical=True, remove_ints=True, drop_na=False)

#well_data = well_data['VSD power frequency']
well_data = well_data['ESP motor temperature']


# Get interval
start_date = pd.Timestamp('2018-01-01')
end_date = pd.Timestamp('2018-08-01')

sub_data = get_sub_sequence(well_data, start_date, end=end_date)
#sub_data = well_data.copy()
sub_data.interpolate(method='linear',inplace=True)
sub_data.dropna(inplace=True)

# fig1, axs1 = print_multiple(sub_data, height=1, width=7)

# sampling_period = 1

# wavelet(sub_data,sampling_period=sampling_period, 
#         period_interval=np.linspace(1, 72, 100),wavelet='cmor12-2')

wavelet = 'db1'


#MPs = get_MP(sub_data.values,subseq_size=24*3)
sub_size = 7*24
ti = perf_counter()
MPs = get_MP_from_wavelets(sub_data,subseq_size=sub_size,level=6,wavelet=wavelet,ignore_start=True, quartile=0.75,create_plot=True,plot_data=True, on_signals=True)
to = perf_counter() -ti
print(f'Time: {to:0.2f} s')


ti = perf_counter()
K, _= get_KDP(sub_data,sub_size,pre_calc_MP=MPs)
to = perf_counter() -ti
print(f'Time: {to:0.2f} s')
print_multiple(K)


# # Define and apply filter
# sampling = 1 # 1 measurement / hour
# #T_cutoff = np.array([30, 6])
# T_cutoff = 2.5
# filter_type = Filter_type.BUTTER_LOW
# order = 2

# cutoff_freq = 1/T_cutoff
# num_coef, den_coef = get_filter(filter_type, sampling_freq=sampling, cutoff_freq=cutoff_freq, order=order)
# filtered_data = apply_filter(well_data, num_coef, den_coef)

# fig2, axs2 = print_multiple(filtered_data, height=1, width=7)

# # Calculate Matrix Profile of well data
# subseq_size = 24
# quartile = 0.75
# MP = get_MP(well_data, subseq_size, quartile=quartile, ignore_start=True)

plt.show()