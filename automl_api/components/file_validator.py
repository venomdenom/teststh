import pandas as pd
from typing import Optional, Callable, Dict

from pandas import DataFrame

from helpers.file_upload_helper import AvailableFileTypes

class FileValidator:
    LOADERS: Dict[AvailableFileTypes: Callable] = {
        AvailableFileTypes.CSV.value: pd.read_csv,
        AvailableFileTypes.EXCEL.value: pd.read_excel,
        AvailableFileTypes.JSON.value: pd.read_json,
        AvailableFileTypes.PARQUET.value: pd.read_parquet,
    }

    def __init__(self, file, file_format: Optional[str] = None) -> None:
        self.file = file
        self.file_format = file_format or self._detect_file_format()

    def _detect_file_format(self):
        extension = self.file.split(".")[-1].lower()
        if extension not in AvailableFileTypes.get_available_file_types():
            raise ValueError(f"File format {extension} is not available")
        return extension

    def _get_loader(self) -> Callable:
        loader = self.LOADERS.get(self.file_format, None)
        if loader is None:
            raise ValueError(f"File format {self.file_format} is not available")
        return loader

    def load(self) -> pd.DataFrame:
        loader = self._get_loader()
        try:
            return loader(file=self.file)
        except Exception as e:
            raise ValueError(f"An error occurred while loading dataset: {e}")

    @staticmethod
    def validate(df: pd.DataFrame) -> bool:
        return False if (df.empty or df.isnull().all().any()) else True

    def process(self) -> DataFrame | None:
        df = self.load()
        if self.validate(df):
            return df
        else:
            return None
