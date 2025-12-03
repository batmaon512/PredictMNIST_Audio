import os
import sys
import json
import pickle
import numpy as np
from flask import Flask, request, jsonify, send_from_directory

# Alias để giải quyết pickle tham chiếu 'modules' và 'modules.HMM'
import HMM as _hmm_module
sys.modules.setdefault('modules', _hmm_module)
sys.modules.setdefault('modules.HMM', _hmm_module)

from HMM import continueHMM
from preprocessing import extract_features
from training import load_model  # dùng hàm load_model từ training.py

# Đường dẫn tuyệt đối
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "hmm_model_1")

# Tải mô hình
models_loaded, scaler, metrics, summary = load_model(MODEL_DIR)
class_names_loaded = summary.get("class_names", [])

app = Flask(__name__, static_folder=os.path.join(BASE_DIR, "web"))

@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.post("/predict")
def predict():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file"}), 400

    file = request.files["audio"]
    data = file.read()

    tmp_path = os.path.join("/tmp", "temp_upload.wav")
    with open(tmp_path, "wb") as f:
        f.write(data)

    mfcc_feats = extract_features(tmp_path)
    if mfcc_feats is None or len(mfcc_feats) == 0:
        return jsonify({"error": "Feature extraction failed"}), 400

    # Chuẩn hóa theo scaler đã lưu
    mfcc_scaled = scaler.transform(mfcc_feats)

    # Tính log-prob mỗi model
    log_probs = []
    for model in models_loaded:
        if model is None:
            log_probs.append(-np.inf)
        else:
            # giả định model.forward trả về (log_prob, ...)
            log_probs.append(model.forward(mfcc_scaled)[0])

    pred_idx = int(np.argmax(log_probs))
    pred_class = class_names_loaded[pred_idx] if class_names_loaded else str(pred_idx)

    lp = np.array(log_probs)
    probs = np.exp(lp - lp.max())
    probs = probs / probs.sum()
    conf = float(probs[pred_idx])

    try:
        os.remove(tmp_path)
    except:
        pass

    return jsonify({
        "label": pred_class,
        "confidence": conf,
        "probs": probs.tolist(),
        "class_names": class_names_loaded
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)