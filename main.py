# ...existing code...
import os, json, pickle
import numpy as np
from flask import Flask, request, jsonify, send_from_directory

from HMM import continueHMM
from preprocessing import extract_features

# Đường dẫn tuyệt đối
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "hmm_model_1")

# Hàm load_model (theo training.py)
def load_model(load_path):
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Không tìm thấy thư mục: {load_path}")

    models_path = os.path.join(load_path, 'models.pkl')
    with open(models_path, 'rb') as f:
        models = pickle.load(f)

    scaler_path = os.path.join(load_path, 'scaler.pkl')
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    metrics_path = os.path.join(load_path, 'metrics.json')
    with open(metrics_path, 'r', encoding='utf-8') as f:
        metrics = json.load(f)
    if 'confusion_matrix' in metrics:
        metrics['confusion_matrix'] = np.array(metrics['confusion_matrix'])
    if 'y_pred' in metrics:
        metrics['y_pred'] = np.array(metrics['y_pred'])

    summary_path = os.path.join(load_path, 'summary.json')
    with open(summary_path, 'r', encoding='utf-8') as f:
        summary = json.load(f)

    return models, scaler, metrics, summary

# Tải mô hình
models_loaded, scaler, metrics, summary = load_model(MODEL_DIR)
class_names_loaded = summary.get("class_names", [])

# Flask app với static folder tuyệt đối
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

    # Render: ghi vào /tmp
    tmp_path = os.path.join("/tmp", "temp_upload.wav")
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
    pred_class = class_names_loaded[pred_idx] if class_names_loaded else str(pred_idx)
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