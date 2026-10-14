from pydantic import BaseModel, Field


class BuildCommit(BaseModel):
    author_id: int = Field(default=42, description="Author ID (1-500)")
    files_changed: int = Field(default=5, ge=1, description="Number of files changed")
    lines_added: int = Field(default=100, ge=0, description="Lines of code added")
    lines_deleted: int = Field(default=20, ge=0, description="Lines of code deleted")
    commit_msg_length: int = Field(default=50, ge=1, description="Character length of commit message")
    hour_of_commit: int = Field(default=14, ge=0, le=23, description="Hour of day (0-23)")
    day_of_week: int = Field(default=2, ge=0, le=6, description="Day of week (0=Monday)")
    branch: str = Field(default="feature", description="Branch name: main, feature, hotfix, release")
    language: str = Field(default="Python", description="Programming language")
    prior_failure_rate: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Author's historical failure rate",
    )


class PredictionResult(BaseModel):
    prediction: str
    confidence: float
    risk_level: str
    top_risk_factors: list[dict]


class LogPredictionRequest(BaseModel):
    log_text: str = Field(..., min_length=1, description="Raw CI/CD build log text")
