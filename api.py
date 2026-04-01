from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io, os, json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "Models")

models = {}
labels = {}

def get_model(crop):
    if crop not in models:
        folder = crop + "Model"
        folder_path = os.path.join(MODELS_DIR, folder)

        model_path = os.path.join(folder_path, f"{crop}_model.keras")
        label_path = os.path.join(folder_path, f"{crop}_labels.json")

        if os.path.exists(model_path):
            models[crop] = load_model(model_path)

        if os.path.exists(label_path):
            with open(label_path) as f:
                labels[crop] = json.load(f)

    return models.get(crop), labels.get(crop)

def preprocess(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

@app.post("/predict/{crop}")
async def predict(crop: str, file: UploadFile = File(...)):
    crop = crop.lower()

    model, label_map = get_model(crop)

    if model is None:
        return {"error": f"{crop} model not found"}

    img_bytes = await file.read()
    img = preprocess(img_bytes)

    preds = model.predict(img)[0]

    class_idx = int(np.argmax(preds))
    confidence = float(np.max(preds))

    disease = label_map[str(class_idx)]

    return {
        "crop": crop,
        "disease": disease,
        "confidence": round(confidence * 100, 2)
    }