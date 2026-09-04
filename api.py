from fastapi import FastAPI, UploadFile, File
import os
import tempfile
import joblib
import pandas as pd

from src.predict_audio import extract_features, load_feature_columns


app = FastAPI(title="Sound2See API")


MODEL_PATH = "models/sound2see_final.joblib"

model = joblib.load(MODEL_PATH)

classifier = model.named_steps.get("classifier")

if classifier is not None and not hasattr(classifier, "multi_class"):
    classifier.multi_class = "auto"


@app.get("/")
def home():
    return {
        "message": "Sound2See API is running"
    }


@app.post("/detect")
async def detect(file: UploadFile = File(...)):

    suffix = os.path.splitext(file.filename)[1] or ".wav"

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as temp:

        temp.write(await file.read())
        audio_path = temp.name

    try:

        features = extract_features(audio_path)

        feature_columns = load_feature_columns()

        X = pd.DataFrame(
            [features],
            columns=feature_columns
        )

        prediction = model.predict(X)[0]

        probabilities = model.predict_proba(X)[0]

        confidence = float(max(probabilities)) * 100

        return {
            "type": "sound",
            "sound": str(prediction),
            "confidence": round(confidence, 2)
        }

    finally:

        if os.path.exists(audio_path):
            os.remove(audio_path)
