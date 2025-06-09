import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from typing import Union

from abc import ABC, abstractmethod

from .core import DataSet

class Interpolation:
    LINEAR = 1

class Extrapolation:
    CONSTANT = 1

class InterpolatorInterface(ABC):

    @property
    @abstractmethod
    def extrapolation_method():
        pass

    @property
    @abstractmethod
    def interpolation_method():
        pass

    @abstractmethod
    def run_interpolation_extrapolation(data):
        pass

class NumpyInterpolator(InterpolatorInterface):

    last_valid_index=None
    last_index_from_previous=None
    extrapolation_method=Extrapolation.CONSTANT
    interpolation_method=Interpolation.LINEAR
    
    def __init__(self):
        pass

    def run_interpolation_extrapolation(self,data:np.ndarray) ->np.ndarray:

        ndim = data.shape[0]
        total_index = np.arange(data.shape[1])


        if self.last_index_from_previous is None:
            self.last_index_from_previous = np.zeros(ndim,dtype=int)
            self.last_valid_index = np.zeros(ndim,dtype=int)

        for i in range(data.shape[0]):

            indices = np.arange(total_index.size-self.last_valid_index[i])
            local_data = data[i,self.last_valid_index[i]:]

            good_indices_address, bad_indices_address = self.get_indices(local_data[:],i)
            self.get_interpolation(local_data[:],good_indices_address,bad_indices_address,indices,i)
            self.get_extrapolation(local_data[:],good_indices_address,indices,i)

            self.last_index_from_previous[i] = data.shape[1]-1
            self.last_valid_index[i] = total_index[self.last_valid_index[i]:][good_indices_address][-1]

         
    def get_indices(self,data:np.ndarray,i: int) -> Union[np.ndarray, np.ndarray]:

        
        if self.last_index_from_previous[i] == 0:
            starter_address = np.zeros(0,dtype=bool)
            new_start = 0
        else:
            new_start = self.last_index_from_previous[i]-self.last_valid_index[i]+1
            starter_address = np.zeros(self.last_index_from_previous[i]-self.last_valid_index[i]+1,dtype=bool)

            if(np.isfinite(data[0])):
                starter_address[0] = True


        good_indices_address = np.append(starter_address,
                                         np.isfinite(data[new_start:]))
                                        
        bad_indices_address = np.append(np.logical_not(starter_address),
                                        np.isnan(data[new_start:]))
        
        return good_indices_address, bad_indices_address


    def get_interpolation(self,data:np.ndarray, good_indices_address: np.ndarray, bad_indices_address: np.ndarray, indices: np.ndarray, i: int):

        match self.interpolation_method:

            case Interpolation.LINEAR:
                func = interp1d(indices[good_indices_address], data[good_indices_address], bounds_error=False, kind='linear')

                data[bad_indices_address] = func(indices[bad_indices_address])
                 

    def get_extrapolation(self,data:np.ndarray, good_indices_address: np.ndarray, indices: np.ndarray, i: int):

        match self.extrapolation_method:

            case Extrapolation.CONSTANT:
                good_indices_location = indices[good_indices_address]
                if not good_indices_address[0]:
                    data[:good_indices_location[0]] = data[good_indices_location[0]]
                if not good_indices_address[-1]:
                    data[good_indices_location[-1]:] = data[good_indices_location[-1]]


class DataFrameBuilder:

    def __init__(self, data):
        
        self._dataframe = DataFrame()
        if isinstance(data,pd.DataFrame):
            self._dataframe.dataset = PandasDataSet()
            self._dataframe.interpolator = NumpyInterpolator()
            self._dataframe.dataset.data = data.values.T
            self._dataframe.dataset.index = data.index

        elif isinstance(data, np.ndarray):
            self._dataframe.dataset = NumpyDataSet()
            self._dataframe.interpolator = NumpyInterpolator()
            self._dataframe.dataset.data = data
            self._dataframe.dataset.index = np.arange(data.size,dtype=int)

        else:
            raise RuntimeError("Invalid type for data")
        
    def build(self):
        self._dataframe._build_initial_data()
        return self._dataframe

        
    def set_interpolation(self,interpolation: Interpolation):
        self._dataframe.interpolator.interpolation_method = interpolation

    def set_extrapolation(self,extrapolation: Extrapolation):
        self._dataframe.interpolator.extrapolation_method = extrapolation


class DataFrame:

    interpolator = None
    dataset = None

    def __init__(self):
        pass


    @property
    def data(self):
        return self.dataset.data
    
    @data.setter
    def data(self,new_data:np.ndarray):
        self.dataset.data = new_data

    @property
    def data_as_pandas(self):
        return pd.DataFrame(self.data.T,index=self.dataset.index)

    def _build_initial_data(self):

        old_data = self.dataset.data
        self.interpolator.run_interpolation_extrapolation(old_data[:])


    def insert_data(self, inserted_data):

        if inserted_data.ndim == 1:
            inserted_data = inserted_data[np.newaxis,:]
        elif inserted_data.ndim >2:
            raise RuntimeError("Incorrect size for data")

        self.dataset.insert(inserted_data)

        old_data = self.dataset.data
        self.interpolator.run_interpolation_extrapolation(old_data[:])


class pyDataSet(DataSet):
    # Inherits .data, .insert_data, .shape, .size, .ndim

    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def index(self):
        pass

    @abstractmethod
    def insert(self,new_data):
        pass



class NumpyDataSet(pyDataSet):

    index = np.arange(0)

    def __init__(self):
        super().__init__()

    def insert(self, new_data: np.ndarray):
        self.index = np.append(self.index,
                               self.size+np.arange(new_data.size))
        self.insert_data(new_data)

class PandasDataSet(pyDataSet):

    index = np.arange(0)

    def __init__(self):
        super().__init__()

    def insert(self, new_data: pd.DataFrame):
        self.index = np.append(self.index,
                               new_data.index)
        self.insert_data(new_data.values.T)
