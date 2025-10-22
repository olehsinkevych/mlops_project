import os
import joblib
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline


class Trainer:
    def __init__(self):
        self.config = self.load_config()
        self.model_name: str = self.config['model']['name']
        self.model_params = self.config['model']['params']
        self.model_path = self.config['model']['store_path']
        self.pipeline = self.create_pipeline()

    @staticmethod
    def load_config():
        with open('config.yaml', 'r') as config_file:
            return yaml.safe_load(config_file)

    def create_pipeline(self):
        preprocessor = ColumnTransformer(
            transformers=[
                ('minmax', MinMaxScaler(), ['AnnualPremium']),
                ('standardize', StandardScaler(), ['Age', ]),
                ('onehot', OneHotEncoder(handle_unknown='ignore'), ['Gender', ])
            ]
        )

        model_map = {
            "RandomForestClassifier": RandomForestClassifier,
            'GradientBoostingClassifier': GradientBoostingClassifier,
        }

        model_class = model_map[self.model_name]
        model = model_class(**self.model_params)

        pipeline = Pipeline(
            ('preprocessor', preprocessor),
            ('model', model)
        )
        return pipeline

    @staticmethod
    def feature_target_separator(data: pd.DataFrame):
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        return X, y

    def train_model(self, X_train, y_train):
        self.pipeline.fit(X_train, y_train)

    def save_model(self):
        model_file_path = os.path.join(
            self.model_path, 'model.pkl'
        )
        joblib.dump(self.pipeline, model_file_path)
