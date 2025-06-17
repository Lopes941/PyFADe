import pandas
import numpy as np
from scipy.interpolate import interp1d
from typing import Self

from abc import ABC, abstractmethod

from .core import DataSet

class Interpolation:
    LINEAR = 1

class Extrapolation:
    CONSTANT = 1

# ========================
# Interpolator class
# ========================

class InterpolatorInterface(ABC):
    """
    Interface for Interpolator objects.

    Properties
    ----------
    extrapolation_method: Extrapolation
        Extrapolation method identifier.
    interpolation_method: Interpolation
        Interpolation method identifier.
    """

    @property
    @abstractmethod
    def extrapolation_method():
        """
        Extrapolation: Extrapolation method
        """
        pass

    @property
    @abstractmethod
    def interpolation_method():
        """
        Interpolation: Interpolation method
        """
        pass

    @abstractmethod
    def run_interpolation_extrapolation(data):
        """
        Runs the interpolation and extrapolation schemes.

        Parameters
        ----------
        data: Data to be interpolated.
        """
        pass


class NumpyInterpolator(InterpolatorInterface):
    """
    Interpolator object for interpolating numpy datasets.

    Properties
    ----------
    extrapolation_method: Extrapolation
        Extrapolation method identifier.
    interpolation_method: Interpolation
        Interpolation method identifier.
    last_index_from_previous: int

    last_valid_index: int
    """
    
    last_valid_index=None
    last_index_from_previous=None
    extrapolation_method=None
    interpolation_method=None

    def __init__(self):
        """
        Initializes the NumpyInterpolator object.
        """
         
        self.extrapolation_method=Extrapolation.CONSTANT
        self.interpolation_method=Interpolation.LINEAR
        self.last_valid_index=None
        self.last_index_from_previous=None

    def run_interpolation_extrapolation(self, data : np.ndarray):
        """
        Runs the interpolation and extrapolation methods on given data

        Parameters
        ----------
        data: np.ndarray
            Data to be interpolated. It is operated by reference.
        """

        ndim = data.shape[0]
        total_index = np.arange(data.shape[1])


        if self.last_index_from_previous is None:
            self.last_index_from_previous = np.zeros(ndim,dtype=int)
            self.last_valid_index = np.zeros(ndim,dtype=int)

        for dim in range(data.shape[0]):

            indices = np.arange(total_index.size-self.last_valid_index[dim])
            local_data = data[dim,self.last_valid_index[dim]:]

            good_indices_address, bad_indices_address = self.get_indices(local_data[:],dim)
            self.get_interpolation(local_data[:],good_indices_address,bad_indices_address,indices,dim)
            self.get_extrapolation(local_data[:],good_indices_address,indices,dim)

            self.last_index_from_previous[dim] = data.shape[1]-1
            self.last_valid_index[dim] = total_index[self.last_valid_index[dim]:][good_indices_address][-1]

         
    def get_indices(self,data:np.ndarray, dim: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Gets the indices that must be interpolated/extrapolated.

        Parameters
        ----------
        data: np.ndarray
            Data to be interpolated. It is returned by reference.
        dim: int
            Current dimension index.

        Returns
        -------
        good_indices_address: np.ndarray
            Boolean array identifying indices that are not NaN.
        bad_indices_address: np.ndarray
            Boolean array identifying indices that are NaN.
        """

        
        if self.last_index_from_previous[dim] == 0:
            starter_address = np.zeros(0,dtype=bool)
            new_start = 0
        else:
            new_start = self.last_index_from_previous[dim]-self.last_valid_index[dim]+1
            starter_address = np.zeros(self.last_index_from_previous[dim]-self.last_valid_index[dim]+1,dtype=bool)

            if(np.isfinite(data[0])):
                starter_address[0] = True


        good_indices_address = np.append(starter_address,
                                         np.isfinite(data[new_start:]))
                                        
        bad_indices_address = np.append(np.logical_not(starter_address),
                                        np.isnan(data[new_start:]))
        
        return good_indices_address, bad_indices_address


    def get_interpolation(self,
                          data:np.ndarray, 
                          good_indices_address: np.ndarray, 
                          bad_indices_address: np.ndarray, 
                          indices: np.ndarray, 
                          dim: int):
        """
        Runs the interpolation scheme on data.

        Parameters
        ----------
        data: np.ndarray
            Data to be interpolated. It is operated by reference.
        good_indices_address: np.ndarray
            Boolean array identifying indices that are not NaN.
        bad_indices_address: np.ndarray
            Boolean array identifying indices that are NaN.
        indices: np.ndarray
            Array with indices to be interpolated.
        dim: int
            Current dimension index.
        """

        match self.interpolation_method:

            case Interpolation.LINEAR:
                func = interp1d(indices[good_indices_address], data[good_indices_address], bounds_error=False, kind='linear')

                data[bad_indices_address] = func(indices[bad_indices_address])
                 

    def get_extrapolation(self, 
                          data:np.ndarray, 
                          good_indices_address: np.ndarray, 
                          indices: np.ndarray, 
                          dim: int):
        """
        Runs the extrapolation scheme on data.

        Parameters
        ----------
        data: np.ndarray
            Data to be interpolated. It is operated by reference.
        good_indices_address: np.ndarray
            Boolean array identifying indices that are not NaN.
        bad_indices_address: np.ndarray
            Boolean array identifying indices that are NaN.
        indices: np.ndarray
            Array with indices to be interpolated.
        dim: int
            Current dimension index.
        """

        match self.extrapolation_method:

            case Extrapolation.CONSTANT:
                good_indices_location = indices[good_indices_address]
                if not good_indices_address[0]:
                    data[:good_indices_location[0]] = data[good_indices_location[0]]
                if not good_indices_address[-1]:
                    data[good_indices_location[-1]:] = data[good_indices_location[-1]]

# ========================
# DataFrame class
# ========================

class DataFrame:
    """Builder pattern that builds a DataFrame object.

    Attributes
    ----------
    dataset: DataSet
        Object that holds the multi-dimensional time series.
    interpolator: InterpolatorInterface
        Object that holds the interpolation and extrapolation methods.

    Properties
    ----------
    data: np.ndarray
        Multi-dimensional time series held by dataset.
    index: np.ndarray
        Index from the multi-dimensional time series held by dataset.
    as_pandas: pandas.DataFrame
        Data and index from dataset, returned as a pandas Dataframe.
    """

    interpolator = None
    dataset = None

    def __init__(self):
        """Initializes the DataFrame object.
        """

        self.dataset = None
        self.interpolator = None

    @property
    def data(self) -> np.ndarray:
        """
        np.ndarray : complete multi-dimensional time series.
        """
        return self.dataset.data
    
    @data.setter
    def data(self,new_data:np.ndarray):
        """
        Sets the multi-dimensional time series value.

        Only call this append new data.

        Parameters
        ----------
        new_data : np.ndarray
            The new data to assign
        """
        self.dataset.data = new_data

    @property
    def index(self) -> np.ndarray:
        """
        np.ndarray : Timestep index from time-series.
        """
        return self.dataset.index

    @property
    def as_pandas(self) -> pandas.DataFrame:
        """
        pandas.DataFrame : Returns the DataFrame as a pandas DataFrame.
        """
        return pandas.DataFrame(self.data.T,index=self.dataset.index)


    def _build_initial_data(self):
        """
        Builds the initial dataset by running the interpolator in the dataset.
        """

        old_data = self.dataset.data
        self.interpolator.run_interpolation_extrapolation(old_data[:])


    def insert_data(self, inserted_data: np.ndarray | pandas.DataFrame):
        """
        Inserts new data in the timeseries. Already runs the interpolators.

        Parameters
        ----------
        inserted_data : np.ndarray or pandas.DataFrame
            New inserted data.
        """

        if isinstance(inserted_data,pandas.DataFrame) and isinstance(self.dataset,PandasDataSet):
            new_data = inserted_data.values
        elif isinstance(inserted_data,np.ndarray) and isinstance(self.dataset,NumpyDataSet):
            new_data = inserted_data
        else:
            raise ValueError("Incorrect type for inserted data")

        if new_data.ndim == 1:
            new_data = new_data[np.newaxis,:]
        elif new_data.ndim >2:
            raise RuntimeError("Incorrect size for data")

        old_data = self.dataset.data
        old_data = np.append(old_data,new_data,axis=1)
        self.interpolator.run_interpolation_extrapolation(old_data[:])

        if isinstance(inserted_data,np.ndarray):
            new_data = old_data[:,-new_data.shape[1]:]
        elif isinstance(inserted_data,pandas.DataFrame):
            new_data = pandas.DataFrame(old_data[:,-new_data.shape[1]:],
                                        index= inserted_data.index)
            
        self.dataset.insert(new_data)

# ========================
# DataFrameBuilder class
# ========================

class DataFrameBuilder:
    """Builder pattern that builds a DataFrame object.
    """

    def __init__(self, data):
        """
        Initializes the DataFrameBuilder object with given data.

        Creates a DataFrame object with given dataset, choosing the appropriate
        derived DataFrame object based on the given data type.

        Parameters
        ----------
        data : np.ndarray | pandas.DataFrame
            Input values to predict.
        """

        self._dataframe = DataFrame()
        if isinstance(data,pandas.DataFrame):
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
        
    def build(self) -> DataFrame:
        """
        Builds the DataFrame object and returns it.

        Parameters
        ----------
        x : ndarray
            Input values to predict.

        Returns
        -------
        dataframe : pyfade.DataFrame
            DataFrame object with the properties defined by the DataFrameBuilder.
        """

        self._dataframe._build_initial_data()
        return self._dataframe

    def set_interpolation(self,interpolation: Interpolation) -> Self:
        """
        Sets the interpolation method for the DataFrame.

        The interpolation method set here will be used to generate artificial data to ocupy NaN
        entries in the dataset. The method will generate values only for NaNs between valid entries.

        Parameters
        ----------
        interpolation : Interpolation
            Interpolation method.

        Returns
        -------
        self : DataFrameBuilder
            For method chaining.
        """
        self._dataframe.interpolator.interpolation_method = interpolation
        return self

    def set_extrapolation(self,extrapolation: Extrapolation) -> Self:
        """
        Sets the extrapolation method for the DataFrame.

        The extrapolation method set here will be used to generate artificial data to ocupy NaN
        entries in the dataset. The method will generate values only for leading or trailing NaNs.

        Parameters
        ----------
        extrapolation : Extrapolation
            Extrapolation method.

        Returns
        -------
        self : DataFrameBuilder
            For method chaining.
        """

        self._dataframe.interpolator.extrapolation_method = extrapolation
        return self

# ========================
# DataSet derived classes
# ========================

"""
The following derived classes inherits from the pybind11 wrapped class Dataset.

Properties
----------
data: np.ndarray
len: int
ndim: int
shape: tuple(int,int)

Methods
-------
insert_data(np.ndarray)
"""

class pyDataSet(DataSet):
    """
    Class that helps to wrap C++ Dataset to Python. Includes index property.

    Properties
    ----------
    index:
        Timestamp of each element in dataset.

    Methods
    -------
    insert(new_data):
        Inserts new_data into dataset.
    """

    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def index(self):
        """
        np.ndarray : Index of dataset.
        """

        pass

    @abstractmethod
    def insert(self,new_data):
        """
        Inserts new data in dataset. Must call insert_data().

        Parameters
        ----------
        new_data:
            New data to be inserted in dataset.
        """

        pass

class NumpyDataSet(pyDataSet):
    """
    Class that holds numpy style dataset.
    """

    index = None

    def __init__(self):
        """
        Initializes the DataSet object.
        """
        
        self.index = np.arange(0)
        super().__init__()

    def insert(self, new_data: np.ndarray):
        """
        Inserts new data into the DataSet object.

        Parameters
        ----------
        new_data: np.ndarray
            New data to be inserted in dataset.
        """

        self.index = np.append(self.index,
                               self.len+np.arange(new_data.size))
        self.insert_data(new_data)

class PandasDataSet(pyDataSet):
    """
    Class that holds pandas dataframe style dataset.
    """

    index = None

    def __init__(self):
        """
        Initializes the DataSet object.
        """
         
        self.index = np.arange(0)
        super().__init__()

    def insert(self, new_data: pandas.DataFrame):
        """
        Inserts new data into the DataSet object.

        Parameters
        ----------
        new_data: pandas.DataFrame
            New data to be inserted in dataset.
        """

        self.index = np.append(self.index,
                               new_data.index)
        self.insert_data(new_data.values.T)
