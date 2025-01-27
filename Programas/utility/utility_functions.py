import pandas as pd

def clean_database(source:str | pd.DataFrame, column_name:str, value:object, drop_column: bool=True, comparison: str = '=') -> pd.DataFrame:
    """ Creates a pandas DataFrame where all data[column_name] == value are kept.

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
    #data.index = pd.to_datetime(data.index)

    # Removing rows
    if comparison == '=':
        data = data[data[column_name] == value]
    elif comparison == '>':
        data = data[data[column_name] > value]
    elif comparison == '<':
        data = data[data[column_name] < value]
    else:
        raise Exception('comparison must be =, > or <')

    # Dropping columns
    if drop_column:
        data.drop(column_name, axis=1, inplace=True)

    return data


