from pydantic import BaseModel, Field
from typing import Optional

"""
    Input fields for a credit risk prediction request.
    Only the most impactful, human-meaningful fields are exposed here;
    remaining model features are filled with sensible defaults.
"""
class LoanApplicationRequest(BaseModel):
    # Financial
    AMT_INCOME_TOTAL: float = Field(..., description="Applicant's total income")
    AMT_CREDIT: float = Field(..., description="Requested credit/loan amount")
    AMT_ANNUITY: Optional[float] = Field(None, description="Loan annuity (monthly payment)")

    # Demographics
    DAYS_BIRTH: int = Field(..., description="Age in days (negative, e.g. -12000 for ~33 years old)")
    DAYS_EMPLOYED: Optional[int] = Field(None, description="Days employed (negative), null if unemployed")
    CNT_CHILDREN: Optional[int] = Field(0, description="Number of children")

    # Categorical - real-world values, not encoded
    CODE_GENDER: Optional[str] = Field("M", description="M or F")
    NAME_EDUCATION_TYPE: Optional[str] = Field(
        "Secondary / secondary special",
        description="Education level"
    )
    NAME_INCOME_TYPE: Optional[str] = Field("Working", description="Income type")
    NAME_FAMILY_STATUS: Optional[str] = Field("Married", description="Family status")
    FLAG_OWN_CAR: Optional[str] = Field("N", description="Y or N")
    FLAG_OWN_REALTY: Optional[str] = Field("Y", description="Y or N")

    # External credit scores (very high importance in our model)
    EXT_SOURCE_1: Optional[float] = Field(None, description="Normalized external credit score 1")
    EXT_SOURCE_2: Optional[float] = Field(None, description="Normalized external credit score 2")
    EXT_SOURCE_3: Optional[float] = Field(None, description="Normalized external credit score 3")

    class Config:
        json_schema_extra = {
            "example": {
                "AMT_INCOME_TOTAL": 180000,
                "AMT_CREDIT": 500000,
                "DAYS_BIRTH": -12000,
                "DAYS_EMPLOYED": -2000,
                "CODE_GENDER": "F",
                "NAME_EDUCATION_TYPE": "Higher education",
                "EXT_SOURCE_2": 0.65,
                "EXT_SOURCE_3": 0.55
            }
        }

"""Output of a credit risk prediction."""
class PredictionResponse(BaseModel):
    default_probability: float = Field(..., description="Probability of default (0-1)")
    risk_category: str = Field(..., description="Low, Medium, or High risk classification")
    prediction: int = Field(..., description="0 = predicted no default, 1 = predicted default")