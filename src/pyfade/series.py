import pandas
import numpy as np
from scipy.interpolate import interp1d
from typing import Self

from abc import ABC, abstractmethod

from .core import DataSet

class Interpolation:
    """
    Interpolation methods available for use in the InterpolatorInterface.

    This class defines the available interpolation methods as constants.

    Attributes
    ----------
    LINEAR : int
        Linear interpolation method identifier.
    """

    LINEAR = 1

class Extrapolation:
    """
    Extrapolation methods available for use in the InterpolatorInterface.

    This class defines the available extrapolation methods as constants.

    Attributes
    ----------
    CONSTANT : int
        Constant extrapolation method identifier.
    """

    CONSTANT = 1

# ========================
# Interpolator class
# ========================

class InterpolatorInterface(ABC):
    """
    Interpolator interface for interpolating and extrapolating data.

    This interface defines the methods and properties that any interpolator
    must implement.
    
    Properties
    ----------
    extrapolation_method: Extrapolation
        Extrapolation method identifier.
    interpolation_method: Interpolation
        Interpolation method identifier.

    Methods
    -------
    run_interpolation_extrapolation(data: np.ndarray):
        Runs the interpolation and extrapolation schemes on the given data.
        Parameters
        ----------
        data: np.ndarray
            Data to be interpolated.

    See Also
    --------
    Interpolation: Enum for interpolation methods.
    Extrapolation: Enum for extrapolation methods.
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
        Runs the interpolation and extrapolation methods on given data.

        Parameters
        ----------
        data: np.ndarray
            Data to be interpolated. It is operated by reference.
        """
        pass


class NumpyInterpolator(InterpolatorInterface):
    """
    NumpyInterpolator class that implements the InterpolatorInterface.
    
    This class provides methods for interpolating and extrapolating data using numpy and scipy.
    It uses linear interpolation and constant extrapolation by default.

    Attributes
    ----------
    extrapolation_method: Extrapolation
        Extrapolation method identifier.
    interpolation_method: Interpolation
        Interpolation method identifier.

    Methods
    -------
    run_interpolation_extrapolation(data: np.ndarray):
        Runs the interpolation and extrapolation methods on given data.
        Parameters
        ----------
        data: np.ndarray
            Data to be interpolated. It is operated by reference.

    See Also
    --------
    InterpolatorInterface: Interface for interpolators that implement interpolation and extrapolation methods.
    Interpolation: Enum for interpolation methods.
    Extrapolation: Enum for extrapolation methods.
    """

    _last_valid_index=None
    _last_index_from_previous=None
    _extrapolation_method=None
    _interpolation_method=None

    def __init__(self):
        """
        Initializes the NumpyInterpolator object.

        Sets the extrapolation and interpolation methods to default values.
        Extrapolation is set to CONSTANT and interpolation is set to LINEAR.
        """
         
        self.extrapolation_method=Extrapolation.CONSTANT
        self.interpolation_method=Interpolation.LINEAR
        self._last_valid_index=None
        self._last_index_from_previous=None

    def run_interpolation_extrapolation(self, data : np.ndarray):
        """
        Runs the interpolation and extrapolation methods on given data.

        Parameters
        ----------
        data: np.ndarray
            Data to be interpolated. It is operated by reference.
            The data must be a 2D array where the first dimension is the number of dimensions
            and the second dimension is the number of time steps.
        """

        ndim = data.shape[0]
        total_index = np.arange(data.shape[1])


        if self._last_index_from_previous is None:
            self._last_index_from_previous = np.zeros(ndim,dtype=int)
            self._last_valid_index = np.zeros(ndim,dtype=int)

        for dim in range(data.shape[0]):

            indices = np.arange(total_index.size-self._last_valid_index[dim])
            local_data = data[dim,self._last_valid_index[dim]:]

            good_indices_address, bad_indices_address = self.get_indices(local_data[:],dim)
            self.get_interpolation(local_data[:],good_indices_address,bad_indices_address,indices,dim)
            self.get_extrapolation(local_data[:],good_indices_address,indices,dim)

            self._last_index_from_previous[dim] = data.shape[1]-1
            self._last_valid_index[dim] = total_index[self._last_valid_index[dim]:][good_indices_address][-1]

         
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
        tuple[np.ndarray, np.ndarray]
            A tuple containing two boolean arrays:
            - good_indices_address: Boolean array identifying indices that are not NaN.
            - bad_indices_address: Boolean array identifying indices that are NaN.
        """

        
        if self._last_index_from_previous[dim] == 0:
            starter_address = np.zeros(0,dtype=bool)
            new_start = 0
        else:
            new_start = self._last_index_from_previous[dim]-self._last_valid_index[dim]+1
            starter_address = np.zeros(self._last_index_from_previous[dim]-self._last_valid_index[dim]+1,dtype=bool)

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
    """
    Builder pattern that builds a DataFrame object.

    This class is used to manage multi-dimensional time series data, allowing for
    interpolation and extrapolation of missing values. It provides properties to access
    the data, index, and a pandas DataFrame representation of the data.

    It is designed to be used in conjunction with the DataFrameBuilder class, which
    simplifies the creation of DataFrame objects by allowing method chaining to set
    various properties.

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

    See Also
    --------
    DataFrameBuilder: Builder class for creating a DataFrame object.
    pyDataSet: Abstract class that holds a dataset.
    NumpyDataSet: Class that holds a dataset in the form of a numpy array.
    PandasDataSet: Class that holds a dataset in the form of a pandas DataFrame.
    InterpolatorInterface: Interface for interpolators that implement interpolation and extrapolation methods.
    Interpolation: Enum for interpolation methods.
    Extrapolation: Enum for extrapolation methods.

    """

    interpolator = None
    dataset = None

    def __init__(self):
        """
        Initializes the DataFrame object.
        """
        self.dataset = None
        self.interpolator = None

    @property
    def data(self) -> np.ndarray:
        """
        np.ndarray : complete multi-dimensional time series.
        """

        if self.dataset is None:
            raise RuntimeError("Dataset must be set before getting data")


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

        if self.dataset is None:
            raise RuntimeError("Dataset must be set before setting data")
        
        if self.interpolator is None:
            raise RuntimeError("Interpolator must be set before setting data")
        
        if self.dataset.data is None:
            raise RuntimeError("Dataset data must be set before setting data")

        if not isinstance(new_data, np.ndarray):
            raise ValueError("Data must be a numpy ndarray")
            
        
        if new_data.ndim > 2:
            raise ValueError("Data must be 1D or 2D array")
        
        if new_data.ndim == 1:
            new_data = new_data[np.newaxis,:]
        if isinstance(self.dataset, PandasDataSet):
            self.dataset.data = pandas.DataFrame(new_data)
        elif isinstance(self.dataset, NumpyDataSet):
            self.dataset.data = new_data

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

        This method is called after the DataFrameBuilder has set the initial data.

        It initializes the dataset with the data provided to the DataFrameBuilder
        and runs the interpolation and extrapolation methods on the initial data.

        It is expected that the dataset has been set before calling this method.
        Raises
        -------
        RuntimeError: If the dataset is not set before calling this method.
        ValueError: If the dataset is not of type NumpyDataSet or PandasDataSet.
        """

        if self.dataset is None:
            raise RuntimeError("Dataset must be set before building the DataFrame")
        
        if not isinstance(self.dataset, (NumpyDataSet, PandasDataSet)):
            raise ValueError("Dataset must be of type NumpyDataSet or PandasDataSet")
        if self.interpolator is None:
            raise RuntimeError("Interpolator must be set before building the DataFrame")
        if not isinstance(self.interpolator, InterpolatorInterface):
            raise ValueError("Interpolator must be of type InterpolatorInterface")
        if self.dataset.data is None:
            raise RuntimeError("Dataset data must be set before building the DataFrame")
        if self.dataset.data.ndim > 2:
            raise RuntimeError("Dataset data must be 1D or 2D array")
        if self.dataset.data.ndim == 1:
            self.dataset.data = self.dataset.data[np.newaxis,:]
        if isinstance(self.dataset.data, pandas.DataFrame):
            self.dataset.data = self.dataset.data.values.T
        if self.dataset.data.ndim != 2:
            raise RuntimeError("Dataset data must be 1D or 2D array")

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
    """
    Builder pattern that builds a DataFrame object.

    This class is used to create a DataFrame object with a given dataset,
    allowing for method chaining to set various properties such as interpolation
    and extrapolation methods. It simplifies the creation of DataFrame objects
    by providing a fluent interface.

    Args
    ----
    data (np.ndarray | pandas.DataFrame): The data to be used in the DataFrame. 

    Attributes
    ----------
    _dataframe : DataFrame
        The DataFrame object being built.

    Methods
    ------- 
    build() -> DataFrame:
        Builds the DataFrame object and returns it.
    set_interpolation(interpolation: Interpolation) -> Self:
        Sets the interpolation method for the DataFrame.
    set_extrapolation(extrapolation: Extrapolation) -> Self:
        Sets the extrapolation method for the DataFrame.

    See Also
    --------
    DataFrame: The DataFrame object that is being built.
    DataSet: Base class for datasets that holds the data and index.
    InterpolatorInterface: Interface for interpolators that implement interpolation and extrapolation methods.
    Interpolation: Enum for interpolation methods.
    Extrapolation: Enum for extrapolation methods.

    Example
    -------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pyfade import DataFrameBuilder, Interpolation, Extrapolation

    >>> data = np.array([[1, 2, np.nan], [4, np.nan, 6]])
    >>> builder = DataFrameBuilder(data)
    >>> dataframe = builder.set_interpolation(Interpolation.LINEAR) \
                            .set_extrapolation(Extrapolation.CONSTANT) \
                            .build()
    >>> print(dataframe.data)
    [[ 1.  2.  2.]
        [ 4.  5.  6.]]
    >>> print(dataframe.as_pandas)
    A    B
    0  1.0  4.0
    1  2.0  5.0
    2  2.0  6.0
    """

    def __init__(self, data : np.ndarray | pandas.DataFrame | pandas.Series):
        """
        Initializes the DataFrameBuilder object with given data.

        Creates a DataFrame object with given dataset, choosing the appropriate
        derived DataFrame object based on the given data type.

        Parameters
        ----------
        data : np.ndarray | pandas.DataFrame | pandas.Series
            The data to be used in the DataFrame. It can be a 1D or 2D numpy array,
            a pandas DataFrame, or a pandas Series. The data will be used to initialize
            the DataFrame object. If a pandas Series is provided, it will be converted 
            to a 2D array.

        Raises
        -------
        RuntimeError: If the data type is not supported.
        ValueError: If the data is not a 1D or 2D array.
        """

        self._dataframe = DataFrame()
        if isinstance(data,pandas.DataFrame):
            self._dataframe.dataset = PandasDataSet()
            self._dataframe.interpolator = NumpyInterpolator()
            self._dataframe.dataset.data = data.values.T
            self._dataframe.dataset.index = data.index

        elif isinstance(data,pandas.Series):
            self._dataframe.dataset = PandasDataSet()
            self._dataframe.interpolator = NumpyInterpolator()
            self._dataframe.dataset.data = data.values[np.newaxis,:]
            self._dataframe.dataset.index = data.index

        elif isinstance(data, np.ndarray):

            if data.ndim > 2:
                raise RuntimeError("Data must be 1D or 2D array")
            if data.ndim == 1:
                data = data[np.newaxis,:]

            self._dataframe.dataset = NumpyDataSet()
            self._dataframe.interpolator = NumpyInterpolator()
            self._dataframe.dataset.data = data
            self._dataframe.dataset.index = np.arange(data.size,dtype=int)

        else:
            raise RuntimeError("Invalid type for data")
        
    def build(self) -> DataFrame:
        """
        Builds the DataFrame object and returns it.

        This method initializes the DataFrame with the data provided to the DataFrameBuilder
        and runs the interpolation and extrapolation methods on the initial data.
        It is expected that the dataset has been set before calling this method.
        
        Returns
        -------
        DataFrame: The built DataFrame object with the dataset and interpolator set.

        Raises
        -------
        RuntimeError: If the dataset is not set before calling this method.
        """

        if self._dataframe.dataset is None:
            raise RuntimeError("Dataset must be set before building the DataFrame")
        
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

        Raises
        -------
        ValueError: If the interpolation method is not of type Interpolation.

        """

        if not isinstance(interpolation, Interpolation):
            raise ValueError("Interpolation must be of type Interpolation")

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

        Raises
        -------
        ValueError: If the extrapolation method is not of type Extrapolation.
        """

        if not isinstance(extrapolation, Extrapolation):
            raise ValueError("Extrapolation must be of type Extrapolation")

        self._dataframe.interpolator.extrapolation_method = extrapolation
        return self

# ========================
# DataSet derived classes
# ========================


class pyDataSet(DataSet):
    """
    Abstract class that holds a dataset.

    This class is used to define the interface for datasets that can be used
    in the DataFrame class. It provides methods for inserting new data and
    accessing the index of the dataset.

    It is expected that derived classes implement the insert method and the index property.

    Properties
    ----------
    index: np.ndarray
        Index of dataset. Must be implemented by derived classes.
    data: np.ndarray
        Multi-dimensional time series data held by dataset.
    len: int
        Length of the dataset. It is the number of time steps in the dataset.
    ndim: int
        Number of dimensions in the dataset. It is the number of rows in the data array.
    shape: tuple
        Shape of the dataset. It is a tuple containing the number of dimensions and the number of time steps.
        
    Methods
    -------
    insert(new_data):
        Inserts new data in dataset. Must be implemented by derived classes.
    insert_data(new_data: np.ndarray | pandas.DataFrame):
        Inserts new data into the dataset. This method is expected to be called by derived classes after the new data has been processed (e.g., interpolated or extrapolated).
    add_observer(observer: std::shared_ptr[IDataSetObserver]):
        Adds an observer to the dataset. The observer will be notified when the dataset is updated.
    remove_observer(observer: std::shared_ptr[IDataSetObserver]):
        Removes an observer from the dataset. The observer will no longer be notified when the dataset is updated.

    See Also
    --------
    DataSet: Base class for datasets that holds the data and index.
    IDataSetObserver: Interface for observers that want to be notified when the dataset is updated.
    IDataSetObservable: Interface for objects that can be observed by IDataSetObserver.    
    """

    def __init__(self):
        """
        Initializes the pyDataSet object.

        This constructor is expected to be called by derived classes.
        
        It initializes the DataSet object and sets the index to None.
        """

        self.index = None

        super().__init__()

    @property
    @abstractmethod
    def index(self):
        """
        np.ndarray : Index of dataset. Must be implemented by derived classes.
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
    Class that holds numpy array style dataset.

    This class is used to hold a dataset in the form of a numpy array.
    It is derived from the pyDataSet class, which defines the interface for datasets.
    
    Properties
    ----------
    index: np.ndarray
        Index of dataset. It is a 1D numpy array that holds the time steps of the dataset.

    Methods
    -------
    insert(new_data: np.ndarray)
        Inserts new data into the dataset.
        Parameters
        ----------
        new_data: np.ndarray
            New data to be inserted in dataset. It is expected to be a 2D numpy array
            where the first dimension is the number of dimensions and the second dimension is the number of time steps.

    Raises
    -------
    ValueError: If the new_data is not a 1D or 2D numpy array.

    See Also
    --------
    pyDataSet: Abstract class that holds a dataset.
    DataSet: Base class for datasets that holds the data and index.
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

        if not isinstance(new_data, np.ndarray):
            raise ValueError("Data must be a numpy ndarray")

        self.index = np.append(self.index,
                               self.len+np.arange(new_data.size))
        self.insert_data(new_data)

class PandasDataSet(pyDataSet):
    """
    Class that holds pandas dataframe style dataset.

    This class is used to hold a dataset in the form of a pandas DataFrame.
    It is derived from the pyDataSet class, which defines the interface for datasets.

    Properties
    ----------
    index: np.ndarray
        Index of dataset. It is a 1D numpy array that holds the time steps of the dataset.

    Methods
    -------
    insert(new_data: pandas.DataFrame)
        Inserts new data into the dataset.
        Parameters
        ----------
        new_data: pandas.DataFrame
            New data to be inserted in dataset. It is expected to be a pandas DataFrame
            where the index is the time steps and the columns are the dimensions.

    Raises
    -------
    ValueError: If the new_data is not a pandas DataFrame.

    See Also
    --------
    pyDataSet: Abstract class that holds a dataset.
    DataSet: Base class for datasets that holds the data and index.
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

        if not isinstance(new_data, pandas.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")

        self.index = np.append(self.index,
                               new_data.index)
        self.insert_data(new_data.values.T)
