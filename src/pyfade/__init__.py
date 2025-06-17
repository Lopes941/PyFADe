import os
import warnings

try:
    from .cuda_config import cuda_path
    os.add_dll_directory(os.path.join(cuda_path, "bin"))
except Exception as e:
    # warnings.warn(f"CUDA path could not be configured: {e}")
    pass


# from .filter import get_filter, apply_filter, Filter_type
# from .models import znormalize
# from .decomposition import get_signal_decomp
from .plot import plot_multiple, plot_shutdown
# from .profile import wavelet_KDP, get_KDP, get_MP, get_MP_from_wavelets, wavelet_MP_from_KDP
# from .utility import clean_database, load_data, get_args, get_sub_sequence, get_well_data, get_shutdowns
# from .series import DataFrameBuilder, Interpolation, Extrapolation
# from .window import WindowBuilder, FeatureGroups

from .series import *
from .window import *