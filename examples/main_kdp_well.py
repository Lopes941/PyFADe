import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

from time import perf_counter
from matplotlib import pyplot as plt
import pyfade

# Source file of data
source_file = 'C:/Users/Heitor/Documents/PosDoc/code/EPIC/Dados/merged_data.csv'

# Load well data
data_full = pyfade.load_data(source_file)
well_data = pyfade.get_well_data(data_full, 'B-18 2', only_numerical=True, remove_ints=True, drop_na=True)


#well_data = well_data['ESP Motor Voltage']
#well_data = well_data['ESP motor temperature']


# Get interval
start_date = pd.Timestamp('2018-01-01')
end_date = pd.Timestamp('2018-08-01')

#sub_data = pyfade.get_sub_sequence(well_data, start_date, end=end_date)
sub_data = well_data.copy()
sub_data.interpolate(method='linear',inplace=True)
#sub_data.dropna(inplace=True)


wavelet = 'db2'

sub_size = 7*24
ti = perf_counter()
MPs = pyfade.wavelet_MP_from_KDP(sub_data,sub_size,3,level=3,wavelet=wavelet,ignore_start=True, quantile=0.75, on_signals=True,create_plot=True,plot_data=True,use_cuda=True)
fig, axs = pyfade.plot_multiple(MPs);
to = perf_counter() -ti
print(f'Time: {to:0.2f} s')


KDP = pyfade.get_KDP(sub_data,sub_size,pre_calc_MP=MPs)
fig, axs = pyfade.plot_multiple(KDP,same_limits=True);

#MP = pyfade.get_MP(well_data, sub_size, quartile=0.75, ignore_start=True)
#fig, axs = pyfade.plot_multiple(MP[0]);

plt.show()