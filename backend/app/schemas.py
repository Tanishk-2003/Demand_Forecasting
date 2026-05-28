from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class DimensionsResponse(BaseModel):
    channels: List[str]
    segments: List[str]
    categories: List[str]
    months: List[str]

class ForecastPoint(BaseModel):
    month: str
    actual: Optional[float] = None
    forecast: Optional[float] = None
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None

class ForecastResponse(BaseModel):
    data: List[ForecastPoint]

class SliceMetric(BaseModel):
    slice_value: str
    mae: float
    mape: float
    smape: float

class MetricsResponse(BaseModel):
    overall: Dict[str, float]
    by_channel: List[SliceMetric]
    by_segment: List[SliceMetric]
    by_category: List[SliceMetric]

class FeatureImportanceItem(BaseModel):
    feature: str
    importance: float

class PerformanceResponse(BaseModel):
    metrics: MetricsResponse
    importance: List[FeatureImportanceItem]

class InsightMetrics(BaseModel):
    overall_mape: float
    overall_smape: float
    total_forecast: float
    growth_percentage: float
    top_category: str
    top_category_share: float
    top_channel: str
    top_channel_share: float

class InsightsResponse(BaseModel):
    ready: bool
    summary: str
    metrics: Optional[InsightMetrics] = None
    bullet_points: List[str]

class RunForecastResponse(BaseModel):
    success: bool
    message: str
