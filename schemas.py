from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict


class ExerciseEnum(str, Enum):
    AUTO = "Auto-detect"
    SQUATS = "Squats"
    PUSHUPS = "Push-ups"
    DOWNWARD_DOG = "Downward Dog"
    WARRIOR = "Warrior Pose"


class ColorCodeEnum(str, Enum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


class FormFeedbackSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text_feedback: List[str] = Field(
        default_factory=list, 
        description="Список текстовых рекомендаций по технике"
    )
    visual_feedback: str = Field(..., description="Краткая плашка состояния для отрисовки")
    color_code: ColorCodeEnum = Field(default=ColorCodeEnum.GREEN)
    voice_message: Optional[str] = Field(default=None)
    form_score: float = Field(..., ge=0.0, le=100.0, description="Оценка техники от 0 до 100")


class FrameLogSchema(BaseModel):
    frame: int = Field(..., ge=0)
    timestamp: float = Field(..., description="Временная метка кадра в секундах")
    exercise: str
    form_feedback: FormFeedbackSchema
    landmarks_detected: bool


class ProcessingJobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class VideoJobResponse(BaseModel):
    job_id: str
    status: ProcessingJobStatus
    processed_video_url: Optional[str] = None
    log_json_url: Optional[str] = None
    error: Optional[str] = None