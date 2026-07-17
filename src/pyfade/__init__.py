import os
import warnings

try:
    from .cuda_config import cuda_path
    os.add_dll_directory(os.path.join(cuda_path, "bin"))
except Exception as e:
    # warnings.warn(f"CUDA path could not be configured: {e}")
    pass


# from .decomposition import get_signal_decomp
from .utility import clean_database, load_data, get_args, get_sub_sequence, get_well_data, get_shutdowns

from . import utility
from . import series
from . import features
from . import group
from . import window

_all = [name for name in dir() if not name.startswith("_") and not name=='core']

__all__ = _all
__all__.extend(series.__all__)
__all__.extend(window.__all__)
__all__.extend(features.__all__)
__all__.extend(group.__all__)

# __all__.extend(analysis.__all__)
# __all__.extend(results.__all__)
# __all__.extend(solver.__all__)