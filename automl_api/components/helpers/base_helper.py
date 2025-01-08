from enum import Enum
from typing import Dict


class TaskType(Enum):
    CLASSIFICATION = 1
    REGRESSION = 2
    CLUSTERING = 3