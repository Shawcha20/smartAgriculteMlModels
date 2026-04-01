from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import tensorflow as tf
import io, os, json
import requests

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

# ======================
# 🔥 MODEL LINKS
# ======================

MODEL_URLS = {
    "mango": {
        "model": "https://drive.google.com/uc?export=download&id=1b_uRB4fiD0xq9UlPTf3Gti7ja9o-4eKI",
        "labels": "https://drive.google.com/uc?export=download&id=1zPSuynFPMHUb0ntLA7Fn9-LNXRQfjAOw"
    },
    "chili": {
        "model": "https://drive.google.com/uc?export=download&id=1P1Cr5iA3WLFqVmD4lxLS3zjHrO6zxVrt",
        "labels": "https://drive.google.com/uc?export=download&id=1fBlfgFihh7BjC_y_vyUdp6fgt-LYwKA4"
    },
    "cabbage": {
        "model": "https://drive.google.com/uc?export=download&id=1lknpT8GYUK0V6EiMkRBLmX38iNoP0CF3",
        "labels": "https://drive.google.com/uc?export=download&id=16FsPpeQhtaWYfCNZL1A_lrIxTQ3UtNBm"
    },
    "brinjal": {
        "model": "https://drive.google.com/uc?export=download&id=1SD8vnvnTRsZ13kfaO5rUoSFb4Ro-RyyM",
        "labels": "https://drive.google.com/uc?export=download&id=1SNZ18j5-a7n8-oA_kq1Uf-HIkW5VvozV"
    }
}

# ======================
# DOWNLOAD FILE
# ======================

def download_file(url, path):
    if not os.path.exists(path):
        print(f"Downloading {path}...")
        r = requests.get(url)
        with open(path, "wb") as f:
            f.write(r.content)

# ======================
# LOAD TFLITE MODEL
# ======================

def load_tflite_model(path):
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter

# ======================
# GET MODEL (LAZY LOAD)
# ======================

def get_model(crop):
    if crop not in models:
        folder_path = os.path.join(MODELS_DIR, crop)
        os.makedirs(folder_path, exist_ok=True)

        model_path = os.path.join(folder_path, f"{crop}.tflite")
        label_path = os.path.join(folder_path, f"{crop}_labels.json")

        if crop in MODEL_URLS:
            download_file(MODEL_URLS[crop]["model"], model_path)
            download_file(MODEL_URLS[crop]["labels"], label_path)

        if os.path.exists(model_path):
            models[crop] = load_tflite_model(model_path)

        if os.path.exists(label_path):
            with open(label_path) as f:
                labels[crop] = json.load(f)

    return models.get(crop), labels.get(crop)

# ======================
# PREPROCESS
# ======================

def preprocess(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0).astype("float32")

# ======================
# PREDICT
# ======================

def predict_tflite(interpreter, img):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])
    return output[0]

# ======================
# API
# ======================

@app.post("/predict/{crop}")
async def predict(crop: str, file: UploadFile = File(...)):
    try:
        crop = crop.lower()

        interpreter, label_map = get_model(crop)

        if interpreter is None:
            return {"error": f"{crop} model not found"}

        img_bytes = await file.read()
        img = preprocess(img_bytes)

        preds = predict_tflite(interpreter, img)

        class_idx = int(np.argmax(preds))
        confidence = float(np.max(preds))

        disease = label_map[str(class_idx)]

        return {
            "crop": crop,
            "disease": disease,
            "confidence": round(confidence * 100, 2)
        }

    except Exception as e:
        return {"error": str(e)}