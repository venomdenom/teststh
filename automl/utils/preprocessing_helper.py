from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder


class DataPreprocessor:
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.data = None

    def load_data(self) -> pd.DataFrame:
        """ Load the data from dataset """
        try:
            self.data = pd.read_csv(self.file_path)
            return self.data
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found at {self.file_path}")
        except Exception as e:
            raise ValueError(f"Error loading file: {e}")

    def _handle_missing_values(self, fill_with_median: bool = False) -> None:
        """ Handle missing values by filling with the median of each column or dropping them """
        if fill_with_median:
            for column in self.data.select_dtypes(include=['float64', 'int64']).columns:
                self.data[column].fillna(self.data[column].median(), inplace=True)
        else:
            self.data.dropna(inplace=True)

    def _encode_categorical(self) -> None:
        """ Encode categorical features using Label Encoding """
        label_encoder = LabelEncoder()
        categorical_columns = self.data.select_dtypes(include=['object']).columns

        for column in categorical_columns:
            self.data[column] = label_encoder.fit_transform(self.data[column])

    def _scale_data(self, scaler_type: str = 'standard') -> None:
        """ Scale numerical features using the selected scaling algorithm """
        match scaler_type:
            case 'standard':
                scaler = StandardScaler()
            case 'minmax':
                scaler = MinMaxScaler()
            case 'robust':
                scaler = RobustScaler()
            case _:
                raise ValueError(
                    f"Scaler type {scaler_type} is not supported. Choose from ['standard', 'minmax', 'robust']")

        numerical_columns = self.data.select_dtypes(include=['float64', 'int64']).columns
        self.data[numerical_columns] = scaler.fit_transform(self.data[numerical_columns])

    def preprocess(self,
                   include_handling_missing_values: bool = True, include_encoding: bool = True,
                   include_data_scaling: bool = True,
                   fill_with_median: bool = False, scaler_type: str = 'standard') -> pd.DataFrame:
        """ Apply all preprocessing steps """
        if include_handling_missing_values:
            self._handle_missing_values(fill_with_median)

        if include_encoding:
            self._encode_categorical()

        if include_data_scaling:
            self._scale_data(scaler_type)

        return self.data

    def split_data(self, target_column: str = 'target', test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """ Split data into train and test sets """
        X = self.data.drop(target_column, axis=1)  # Features
        y = self.data[target_column]  # Target column

        return train_test_split(X, y, test_size=test_size, random_state=42)
