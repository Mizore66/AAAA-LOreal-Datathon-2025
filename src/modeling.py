# Placeholder for modeling utilities (Phase 3)

from dataclasses import dataclass
from typing import Optional
import pandas as pd

@dataclass
class Anomaly:
    feature: str
    timestamp: pd.Timestamp
    score: float
    platform: Optional[str] = None
