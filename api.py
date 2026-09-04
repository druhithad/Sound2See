from fastapi import FastAPI, UploadFile, File
import os
import tempfile
import traceback
import joblib
import pandas as pd

from src.predict_audio import extract_features

app = FastAPI(title="Sound2See API")

MODEL_PATH = "models/sound2see_final.joblib"

try:
    model = joblib.load(MODEL_PATH)

    classifier = model.named_steps.get("classifier")

    if classifier is not None and not hasattr(classifier, "multi_class"):
        classifier.multi_class = "auto"

    print("MODEL LOADED SUCCESSFULLY")

except Exception:
    print("MODEL LOAD ERROR:")
    traceback.print_exc()
    model = None


@app.get("/")
def home():
    return {"message": "Sound2See API is running"}


@app.post("/detect")
async def detect(file: UploadFile = File(...)):

    if model is None:
        return {
            "error": "Model failed to load on server."
        }

    suffix = os.path.splitext(file.filename)[1] or ".wav"

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as temp:

        temp.write(await file.read())
        audio_path = temp.name

    try:

        print("STEP 1: Extracting features")

        features = extract_features(audio_path)

        print("STEP 2: Preparing features")

        # Use the feature names stored inside the trained scaler.
        # This avoids depending on the missing CSV file.
        scaler = model.named_steps.get("scaler")

        if hasattr(scaler, "feature_names_in_"):
            feature_columns = list(scaler.feature_names_in_)
        else:
            feature_columns = list(features.keys())

        X = pd.DataFrame(
            [[features[column] for column in feature_columns]],
            columns=feature_columns
        )

        print("STEP 3: Predicting")

        prediction = model.predict(X)[0]

        print("STEP 4: Predicting probabilities")

        probabilities = model.predict_proba(X)[0]

        confidence = float(max(probabilities)) * 100

        print("SUCCESS:", prediction, confidence)

        return {
            "type": "sound",
            "sound": str(prediction),
            "confidence": round(confidence, 2)
        }

    except Exception as e:

        print("DETECT ERROR:")
        traceback.print_exc()

        return {
            "error": str(e),
            "error_type": type(e).__name__
        }

    finally:

        if os.path.exists(audio_path):
            os.remove(audio_path)
