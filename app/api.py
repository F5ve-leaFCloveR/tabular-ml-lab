from __future__ import annotations

from functools import lru_cache
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from tabular_ml_lab.infer import load_artifacts, predict_proba


class PredictRequest(BaseModel):
    age: int
    workclass: str
    fnlwgt: int
    education: str
    education_num: int = Field(..., alias="education-num")
    marital_status: str = Field(..., alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(..., alias="capital-gain")
    capital_loss: int = Field(..., alias="capital-loss")
    hours_per_week: int = Field(..., alias="hours-per-week")
    native_country: str = Field(..., alias="native-country")

    def to_record(self) -> dict[str, Any]:
        data = self.model_dump(by_alias=True)
        return {
            "age": data["age"],
            "workclass": data["workclass"],
            "fnlwgt": data["fnlwgt"],
            "education": data["education"],
            "education-num": data["education-num"],
            "marital-status": data["marital-status"],
            "occupation": data["occupation"],
            "relationship": data["relationship"],
            "race": data["race"],
            "sex": data["sex"],
            "capital-gain": data["capital-gain"],
            "capital-loss": data["capital-loss"],
            "hours-per-week": data["hours-per-week"],
            "native-country": data["native-country"],
        }


class PredictResponse(BaseModel):
    probability_gt_50k: float


@lru_cache(maxsize=1)
def _artifacts():
    model, preprocessor, metadata = load_artifacts("models")
    return model, preprocessor, metadata


app = FastAPI(title="Tabular ML Lab API", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    model, preprocessor, _ = _artifacts()
    probs = predict_proba([payload.to_record()], model, preprocessor)
    return PredictResponse(probability_gt_50k=float(probs[0]))
