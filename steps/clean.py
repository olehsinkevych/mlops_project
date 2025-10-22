import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

class Cleaner:
    def __init__(self):
        self.imputer = SimpleImputer(
            strategy='most_frequent', missing_values=np.nan
        )

    def clean_data(self, data: pd.DataFrame):
        data.drop(['id'], axis=1, inplace=True)
        # TODO: decide which columns to clean via data analysis in jupyter notebook
        return data