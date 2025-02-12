import pandas as pd
import numpy as np
import warnings
from typing import Literal, get_args
from os.path import join, dirname
from typing import Literal, Tuple, Union, Any


_COMPARISON = Literal['=','<','>']

INTEGER_COLUMNS = ['outlier_',
                   'extValues_',
                   'peaks_',
                   'Well aligned',
                   'Choke',
                   ]

def clean_database(source:str | pd.DataFrame, column_name:str, value:object, drop_column: bool=True, comparison: str = '=') -> pd.DataFrame:
    """ Creates a pandas DataFrame where all data where "data[column_name] == value" are kept.

        Loads the file named source or a DataFrame object, and creates a DataFrame where all rows in which column_name == 0 are removed.

        Parameters
        ----------
        source: str, pandas DataFrame
            Name of the source CSV file or source DataFrame
        column_name: str
            Identifier of the column to be determined
        value: object
            Value to be checked and kept
        drop_column: boolean, optional (default True)
            Boolean to determine whether the column is dropped from the DataFrame
        comparison: '=', '>', '<'
            Type of comparison to be made to determine values to be removed

        Returns
        -------
        database: pandas DataFrame
            DataFrame with the clean data
    """

    # Reading file
    if isinstance(source,pd.DataFrame):
        data = source.copy()
    else:
        data = pd.read_csv(source)
    data.index = pd.to_datetime(data.index)

    # Removing rows
    options = get_args(_COMPARISON)
    assert comparison in options, f"{comparison} is not in {options}"
    if comparison == '=':
        data = data[data[column_name] == value]
    elif comparison == '>':
        data = data[data[column_name] > value]
    elif comparison == '<':
        data = data[data[column_name] < value]

    # Dropping columns
    if drop_column:
        data.drop(column_name, axis=1, inplace=True)

    return data

def load_data(source: str) -> pd.DataFrame:
    """ Loads the dataset.

        Loads the dataset as saved by data_setup.ipynb.

        Parameters
        ----------
        source: str
            Name of the source CSV file
        Returns
        -------
        database: pandas DataFrame
            DataFrame with the clean data
    """
    
    real_path = join(dirname(dirname(__file__)), source)
    data = pd.read_csv(real_path,index_col='time')
    data.index = pd.to_datetime(data.index)
    data.index = data.index.tz_localize(None)
    data['Failure distance'] = pd.to_timedelta(data['Failure distance'])

    return data

def get_well_data(source_data: str | pd.DataFrame, well_name: str, drop_columns: list = [], only_numerical: bool = False, remove_ints: bool = False, drop_na: bool = True, replace_na: float = np.nan) -> pd.DataFrame:
    """ Loads the data from a given well.

        Parameters
        ----------
        source_data: str or pandas DataFrame
            Database. Can be given as a pandas DataFrame or as a string pointing to the source file. In this last case, the database is reloaded.

        well_name: str
            Well run name

        drop_columns: list, optional
            List containing columns to be dropped by name. All columns containing this name in some way are removed. For instance, if 'outlier_' is given, all columns that contain 'outlier_' in their name will be dropped.

        only_numerical: bool, optional (default False)
            If True, drops all non-numerical columns from the database.

        remove_ints: bool, optional (default False)
            If True, drops all columns that are known to be composed of integers.

        drop_na: bool, optional (default True)
            If True, drops all columns full of only NAs
        
        replace_na: float, optional (default np.nan)
            Replace all NAs with this value if drop_na is true.
        
        Returns
        -------
        well_data: pandas DataFrame
            DataFrame with the data from specified well
    """

    # Loading database if input is string
    if isinstance(source_data,str):
        source_data = load_data(source_data)

    # Getting well data
    well_data = source_data[source_data['Well Run'] == well_name]

    # Dropping non-numerical
    if only_numerical:
        well_data = well_data.select_dtypes(include=[float])

    # Dropping counted
    if remove_ints:
        for int_col in INTEGER_COLUMNS:
            well_data.drop(list(well_data.filter(regex=int_col)), inplace=True,axis=1)

    # Dropping given columns
    for drop_col in drop_columns:
        well_data.drop(list(well_data.filter(regex=drop_col)), inplace=True, axis=1)

    # Dropping Na columns
    if drop_na:
        well_data.dropna(axis=1, how='all', inplace=True)

    if replace_na != np.nan:
        well_data.fillna(replace_na, inplace=True)
        
    return well_data

def get_sub_sequence(data: Union[pd.DataFrame, pd.Series, np.ndarray], start, end = None, size: int = 0, return_gaps: bool = False) -> Union[Any, Tuple[ Any, list]]:
    """ Gets a subsequence of a time series based on the given start and end or size.

        
        Parameters
        ----------
        data: pandas.DataFrame or pandas.Series or numpy.ndarray
            Set of data from which sub_sequences will be generated.
            In case of multi-dimensional time series given as a numpy ndarray, its dimensions must be:
            data.size == (num_dim, series_size)
        start: int, index of pandas.DataFrame
            Start point from the returned subsequence.
        end: int, index of pandas.DataFrame
            End point from the returned subsequence, end point not included!
        size: int, optional
            Size of the subsequence.
        return_gaps: bool, optional (default = False)
            Changes the return variables to include the gaps in this signal.

        Returns
        -------
        sub_data: pandas.DataFrame or pandas.Series or numpy.ndarray
            Obtained subsequence. Will be of same type as input data.
        gaps: 
            L
    """

    # Checking size and end inputs
    if size == 0 and end is None:
        raise Exception('Either size or end must be given!')
    if size != 0 and end is not None:
        if start + size != end:
            warnings.warn('Both size and end were given, but they are not consistent, end variable used.')

    # Calculating end if necessary
    if end is None:
        end = start + size
    
    if isinstance(data, np.ndarray):

        if data.ndim == 1:
            sub_data = data[np.newaxis, start:end]
        else:
            sub_data = data[:, start:end]


    else:
    
        sub_data = data.loc[start:end]


    return sub_data

