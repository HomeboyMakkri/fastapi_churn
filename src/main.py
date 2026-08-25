from fastapi import FastAPI

from .schemas import FeatureVectorChurn

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "ml churn server is running"}

@app.post("/predict", response_model=FeatureVectorChurn)
def predict_churn(feature_vector: FeatureVectorChurn) -> FeatureVectorChurn:
    return feature_vector