from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class AnalysisType(str, Enum):
    PLANT_PEST = "PLANT_PEST"
    SATELLITE_REMOTE_SENSING = "SATELLITE_REMOTE_SENSING"
    DRONE_ANALYSIS = "DRONE_ANALYSIS"


class SeverityLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class BoundingBox(BaseModel):
    x_min: float = Field(..., ge=0.0, le=1.0)
    y_min: float = Field(..., ge=0.0, le=1.0)
    x_max: float = Field(..., ge=0.0, le=1.0)
    y_max: float = Field(..., ge=0.0, le=1.0)


class SolutionDetail(BaseModel):
    solution: str
    details: str
    source: Optional[str] = None


class RecommendationsGroup(BaseModel):
    biological: List[SolutionDetail]
    chemical: List[SolutionDetail]
    cultural: List[SolutionDetail]


class DetailedInfo(BaseModel):
    description: str
    impact: str
    recommendations: RecommendationsGroup
    knowledgeBaseTags: List[str]


class Detection(BaseModel):
    className: str
    confidenceScore: float
    severity: SeverityLevel
    boundingBox: BoundingBox
    details: DetailedInfo
    croppedImageUrl: Optional[str] = Field(None)


class AnalysisSubject(BaseModel):
    subjectType: str
    description: str
    confidence: float


class AIAnalysisResponse(BaseModel):
    subject: AnalysisSubject
    detections: List[Detection]
