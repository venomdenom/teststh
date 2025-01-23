from enum import Enum
from typing import List


class AvailableFileTypes(Enum):
    CSV = 'csv'
    JSON = 'json'
    EXCEL = 'xlsx'
    PARQUET = 'parquet'

    @classmethod
    def get_available_file_types(cls) -> List[str]:
        return list(map(lambda x: x.value, cls))