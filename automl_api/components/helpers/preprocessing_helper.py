from typing import Dict, Callable
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from enum import Enum
from dataclasses import dataclass

class ScaleType(Enum):
    STANDARD = 1
    MINMAX = 2
    ROBUST = 3

class ScaleTypes:
    scalers: Dict[ScaleType, Callable] = {
        ScaleType.STANDARD: StandardScaler,
        ScaleType.MINMAX: MinMaxScaler,
        ScaleType.ROBUST: RobustScaler,
    }

    @classmethod
    def get_scaler(cls, scale_type: ScaleType) -> Callable:
        scaler_class = cls.scalers.get(scale_type, None)
        if not scaler_class:
            raise ValueError(f"Scale type {scale_type} not supported")

        return scaler_class

@dataclass
class PreprocessingParams:
    include_handling_missing_values: bool = True
    include_feature_encoding: bool = True
    include_data_scaling: bool = True
    fill_with_median: bool = False
    scale_type: ScaleType = ScaleType.STANDARD