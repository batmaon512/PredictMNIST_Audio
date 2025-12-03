# ...existing code...
import os, json, pickle
import numpy as np
from flask import Flask, request, jsonify, send_from_directory

from HMM import continueHMM
from preprocessing import extract_features
# ...existing code...

MODEL_DIR = os.path.join("hmm_model_1")

def load_model_from_directory(load_path: str):
    with open(os.path.join(load_path, "models.pkl"), "rb") as f:
        models = pickle.load(f)
    with open(os.path.join(load_path, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(load_path, "summary.json"), "r", encoding="utf-8") as f:
        summary = json.load(f)
    class_names = summary["class_names"]
    return models, class_names, scaler

models_loaded, class_names_loaded, scaler = load_model_from_directory(MODEL_DIR)

app = Flask(__name__, static_folder="web")

@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.post("/predict")
def predict():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file"}), 400

    file = request.files["audio"]
    data = file.read()

    tmp_path = "temp_upload.wav"
    with open(tmp_path, "wb") as f:
        f.write(data)

    mfcc = extract_features(tmp_path)
    if mfcc is None or len(mfcc) == 0:
        return jsonify({"error": "Feature extraction failed"}), 400

    mfcc_scaled = scaler.transform(mfcc)

    log_probs = []
    for model in models_loaded:
        if model is None:
            log_probs.append(-np.inf)
        else:
            log_probs.append(model.forward(mfcc_scaled)[0])

    pred_idx = int(np.argmax(log_probs))
    pred_class = class_names_loaded[pred_idx]
    lp = np.array(log_probs)
    probs = np.exp(lp - lp.max()); probs = probs / probs.sum()
    conf = float(probs[pred_idx])

    try: os.remove(tmp_path)
    except: pass

    return jsonify({
        "label": pred_class,
        "confidence": conf,
        "probs": probs.tolist(),
        "class_names": class_names_loaded
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)