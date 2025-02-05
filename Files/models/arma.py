import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import pmdarima
import pywt
from pmdarima import ARIMA
from .preprocessing import _extract_values

from typing import Literal, Tuple, Union, Any

def get_arima(data: Union[pd.DataFrame, pd.Series, np.ndarray], order: Any):

    model = ARIMA(order=order, with_intercept=True, )

    return None

