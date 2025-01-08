from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from .helpers.preprocessing_helper import PreprocessingParams, ScaleType, ScaleTypes

class DataPreprocessor:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def preprocess(self, params: PreprocessingParams):
        """
        :return:
            pd.DataFrame: preprocessed data
        """
        if params.include_handling_missing_values:
            self._handle_missing_values(params.fill_with_median)
        if params.include_feature_encoding:
            self._encode_categorical()
        if params.include_data_scaling:
            self._scale_data(params.scale_type)

        return self.data

    def _handle_missing_values(self, fill_with_median: bool) -> None:
        if fill_with_median:
            for column in self.data.select_dtypes(include=['float64', 'int64']).columns:
                self.data[column].fillna(self.data[column].median(), inplace=True)
        else:
            self.data.dropna(inplace=True)

    def _encode_categorical(self) -> None:
        label_encoder = LabelEncoder()
        categorical_columns = self.data.select_dtypes(include=['object']).columns

        for column in categorical_columns:
            self.data[column] = label_encoder.fit_transform(self.data[column])

    def _scale_data(self, scale_type: ScaleType) -> None:
        scaler_class = ScaleTypes.get_scaler(scale_type)
        scaler = scaler_class()

        numerical_columns = self.data.select_dtypes(include=['float64', 'int64']).columns
        self.data[numerical_columns] = scaler.fit_transform(self.data[numerical_columns])

    def split_data(self, target_column: str = 'target', test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train and test sets
        :param target_column:
            Name of target column
        :param test_size:
            Proportion of data set to be used for testing
        :param random_state:
            Random state for reproducibility
        :return:
            Split train and test sets
        """
        X = self.data.drop(target_column, axis=1)  # Features
        y = self.data[target_column]  # Target column

        return train_test_split(X, y, test_size=test_size, random_state=random_state)
