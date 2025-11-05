import yaml
from typing import Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class Ingestion:
    def __init__(self, config_path: str) -> None:
        self.config = self._load_config(config_path)

    @staticmethod
    def _load_config(config_path: str):
        with open(config_path, "r") as file:
            return yaml.safe_load(file)

    def load_data(self) -> Tuple:
        train_data_path = self.config['data']['train_path']
        test_data_path = self.config['data']['test_path']
        target = self.config['data']['target']
        train_data = pd.read_csv(train_data_path)
        test_data = pd.read_csv(test_data_path)

        # to split train data onto validation data we can use in-build scikit-learn lib
        X = train_data.drop(target, axis=1)
        y = train_data[target]
        # Split the original data into a main training set and a final test set
        X_train_main, X_test, y_train_main, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        # Split the main training set into a new training set and a validation set
        # For example, allocate 20% of the main training set for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_main, y_train_main, test_size=0.25, random_state=42, stratify=y_train_main
        )
        # TODO: convert to csv from split data by sk-learn
        train_data = pd.DataFrame(data=np.c_[X_train, y_train],
                             columns=X.columns + y.columns)
        val_data = pd.DataFrame(data=np.c_[X_val, y_val],
                             columns=X.columns + y.columns)

        return train_data, val_data, test_data
