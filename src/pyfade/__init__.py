import os

try:
    cuda_home = os.environ['CUDA_PATH']
    os.add_dll_directory(f"{cuda_home}\\bin")
except:
    print("Hi")
    pass


from .filter import get_filter, apply_filter, Filter_type
from .models import get_arima, znormalize
from .decomposition import get_signal_decomp
from .plot import plot_multiple, plot_shutdown
from .profile import wavelet_KDP, get_KDP, get_MP, get_MP_from_wavelets, wavelet_MP_from_KDP
from .utility import clean_database, load_data, get_args, get_sub_sequence, get_well_data, get_shutdowns
from . import mat_profile