from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import os
import json

app = FastAPI()

# ✅ CORS FIX (VERY IMPORTANT)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.getcwd()
MODELS_DIR = os.path.join(BASE_DIR, "Models")

models = {}
labels = {}

# ======================
# LOAD ALL MODELS
# ======================
def load_all_models():
    for folder in os.listdir(MODELS_DIR):
        folder_path = os.path.join(MODELS_DIR, folder)

        if os.path.isdir(folder_path):
            crop = folder.replace("Model", "").lower()

            model_path = os.path.join(folder_path, f"{crop}_model.keras")
            label_path = os.path.join(folder_path, f"{crop}_labels.json")

            if os.path.exists(model_path):
                print(f"Loading {crop} model...")
                models[crop] = load_model(model_path)

            if os.path.exists(label_path):
                with open(label_path) as f:
                    labels[crop] = json.load(f)

load_all_models()

# ======================
# IMAGE PREPROCESS
# ======================
def preprocess(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ======================
# PREDICT API
# ======================
@app.post("/predict/{crop}")
async def predict(crop: str, file: UploadFile):
    crop = crop.lower()

    if crop not in models:
        return {"error": f"{crop} model not found"}

    img_bytes = await file.read()
    img = preprocess(img_bytes)

    preds = models[crop].predict(img)[0]

    class_idx = int(np.argmax(preds))
    confidence = float(np.max(preds))

    disease = labels[crop][str(class_idx)]

    return {
        "crop": crop,
        "disease": disease,
        "confidence": round(confidence * 100, 2)
    }