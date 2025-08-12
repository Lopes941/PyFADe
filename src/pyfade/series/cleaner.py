import numpy as np

class DataCleaner():

    def __init__(self):
        self.column_name = []
        self.value = []
        self.comparison = []
        self.drop_column = []
    
    def add_clean(self, column_name, value: object, comparison: str = '=', drop_column=True):

        self.column_name.append(column_name)
        self.value.append(value)
        self.drop_column.append(drop_column)

        if comparison not in ['=', '>', '<']:
            raise AttributeError("Comparison type must be '=', '>' or '<'")

        self.comparison.append(comparison)

    def run_cleaner(self, data: np.ndarray, index: np.ndarray) -> np.ndarray:

        mask = np.ones(data.shape[0],dtype=bool)
        remove_cols = []

        for i in range(len(self.column_name)):

            col = self.column_name[i]
            val = self.value[i]
            comp = self.comparison[i]

            col_idx = np.where(col==index)[0][0]
            col_data = data[col_idx,:]

            if comp == '=':
                mask &= (col_data == val)
            elif comp == '>':
                mask &= (col_data > val)
            elif comp == '<':
                mask &= (col_data < val)

            if self.drop_column[i]:
                remove_cols.append(col_idx)

        data[:] = data[mask]

        remove_cols = sorted(set(remove_cols), reverse=True)
        data = np.delete(data, remove_cols, axis=1)

        return data
