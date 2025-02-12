"""
Functions for creating a fit for ARIMA and ARIMAX models
"""

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import pmdarima
import pywt
from pmdarima import ARIMA

from typing import Literal, Tuple, Union, Any

#TODO Write ARIMA and ARIMAX functions here
def get_arima(data: Union[pd.DataFrame, pd.Series, np.ndarray], order: Any):

    model = ARIMA(order=order, with_intercept=True, )

    return None

