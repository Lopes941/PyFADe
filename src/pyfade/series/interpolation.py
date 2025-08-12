
from abc import ABC, abstractmethod
import numpy as np
from scipy.interpolate import interp1d

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
    def run_interpolation_extrapolation(self, data):
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
    extrapolation_method=Extrapolation.CONSTANT
    interpolation_method=Interpolation.LINEAR

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
