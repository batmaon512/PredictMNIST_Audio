import os
import json
import pickle
import warnings
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report
from HMM import continueHMM  # tuyệt đối

# Các import dùng để vẽ biểu đồ chỉ phục vụ training, tránh import khi deploy
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:
    plt = None
    sns = None

warnings.filterwarnings("ignore", category=DeprecationWarning)

def _init_params(num_states, seq_list):
    X = np.vstack(seq_list)
    D = X.shape[1]
    pi = np.full(num_states, 1.0/num_states)
    A = np.zeros((num_states, num_states))
    for i in range(num_states):
        stay = 0.6
        move = 0.4
        if i == num_states - 1:
            A[i, i] = 1.0
        else:
            A[i, i] = stay
            A[i, i+1] = move
    A /= A.sum(axis=1, keepdims=True)

    means = np.random.randn(num_states, D)
    global_var = np.var(X, axis=0) + 1e-3
    
    covariances = np.zeros((num_states, D, D))
    for s in range(num_states):
        covariances[s] = np.diag(global_var)

    return A, pi, means, covariances

def train_hmm(X_train, y_train, class_names, num_states=5, n_loop=30, tol=1e-3):
    """
    Huấn luyện mô hình HMM
    
    Args:
        X_train: Dữ liệu huấn luyện
        y_train: Nhãn huấn luyện
        class_names: Tên các lớp
        num_states: Số trạng thái ẩn
        n_loop: Số vòng lặp tối đa
        tol: Ngưỡng hội tụ
    
    Returns:
        models: Dictionary chứa các mô hình HMM đã huấn luyện
    """
    print(f"\n{'='*60}")
    print(f"🚀 BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH HMM")
    print(f"{'='*60}")
    print(f"   - Số trạng thái ẩn: {num_states}")
    print(f"   - Số vòng lặp tối đa: {n_loop}")
    print(f"   - Ngưỡng hội tụ: {tol}")
    print(f"{'='*60}\n")
    
    models = []
    for cls_id, cls_name in enumerate(class_names):
        seq_list = [X_train[i] for i, y in enumerate(y_train) if y == cls_id]
        A, pi, means, covs = _init_params(num_states, seq_list)
        model = continueHMM(A=A, means=means, covariances=covs, pi=pi).fit(seq_list, n_loop=n_loop, bound_learning=tol)
        models.append(model)
        print(f"✅ Hoàn thành huấn luyện lớp: {cls_name}\n")
    
    print(f"{'='*60}")
    print(f"✅ HOÀN THÀNH HUẤN LUYỆN TẤT CẢ CÁC LỚP")
    print(f"{'='*60}\n")
    
    return models


def evaluate_hmm(models, X_test, y_test, class_names):
    """
    Đánh giá mô hình HMM
    """
    print(f"\n{'='*60}")
    print(f"🔍 BẮT ĐẦU ĐÁNH GIÁ MÔ HÌNH")
    print(f"{'='*60}\n")
    
    # Dự đoán
    y_pred = []
    for seq in X_test:
        scores = [m.forward(seq)[0] for m in models]
        y_pred.append(int(np.argmax(scores)))

    # Metrics chung
    acc = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    precision_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    recall_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Ma trận nhầm lẫn
    print(f"{'='*60}")
    print(f"📊 MA TRẬN NHẦM LẪN")
    print(f"{'='*60}")
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix (raw):")
    
    # Vẽ heatmap confusion
    plt.figure(figsize=(max(6, len(class_names)*0.7), max(5, len(class_names)*0.7)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()
    
    # Accuracy từng lớp: đúng / tổng thực tế lớp đó
    per_class_accuracy = []
    supports = cm.sum(axis=1)
    for i in range(len(class_names)):
        acc_i = (cm[i, i] / supports[i]) if supports[i] > 0 else 0.0
        per_class_accuracy.append(acc_i)
    
    print(f"\n{'='*60}")
    print("📌 ACCURACY TỪNG LỚP")
    print(f"{'='*60}")
    print(f"{'Lớp':<20} {'Support':>8} {'Correct':>8} {'Acc':>8}")
    for i, cls in enumerate(class_names):
        print(f"{cls:<20} {supports[i]:>8} {cm[i,i]:>8} {per_class_accuracy[i]:>8.2%}")
    
    # Báo cáo phân loại
    print(f"\n{'='*60}")
    print(f"📋 BÁO CÁO PHÂN LOẠI CHI TIẾT")
    print(f"{'='*60}")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
    
    # Tổng kết
    print(f"{'='*60}")
    print(f"📊 TỔNG KẾT KẾT QUẢ")
    print(f"{'='*60}")
    print(f"   - Accuracy (Global):  {acc:.4f}")
    print(f"   - Precision Macro:    {precision_macro:.4f}")
    print(f"   - Precision Weighted: {precision_weighted:.4f}")
    print(f"   - Recall Macro:       {recall_macro:.4f}")
    print(f"   - Recall Weighted:    {recall_weighted:.4f}")
    print(f"   - F1 Macro:           {f1_macro:.4f}")
    print(f"   - F1 Weighted:        {f1_weighted:.4f}")
    print(f"{'='*60}\n")
    
    metrics = {
        'accuracy': acc,
        'precision_macro': precision_macro,
        'precision_weighted': precision_weighted,
        'recall_macro': recall_macro,
        'recall_weighted': recall_weighted,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'confusion_matrix': cm,
        'y_pred': y_pred,
        'per_class_accuracy': np.array(per_class_accuracy)
    }
    return metrics


def train_and_evaluate_continue_hmm(X_train, X_test, y_train, y_test, class_names, 
                                    num_states=5, n_loop=30, tol=1e-3):
    """
    Hàm kết hợp huấn luyện và đánh giá (giữ lại để tương thích ngược)
    
    Returns:
        models: Dictionary chứa các mô hình HMM
        metrics: Dictionary chứa các chỉ số đánh giá
    """
    # Huấn luyện
    models = train_hmm(X_train, y_train, class_names, num_states, n_loop, tol)
    # Đánh giá
    metrics = evaluate_hmm(models, X_test, y_test, class_names)
    return models, metrics

def save_model(models, scaler, metrics, class_names, save_dir='saved_models', model_name=None):
    """
    Lưu mô hình HMM (list), scaler và metrics.
    
    Args:
        models: List (danh sách) chứa các mô hình HMM đã huấn luyện
        scaler: Scaler đã được fit
        metrics: Dictionary chứa các chỉ số đánh giá
        class_names: List (danh sách) tên các lớp (ví dụ: ['class_A', 'class_B'])
    """
    # 1. Tạo tên mô hình nếu chưa có
    if model_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"hmm_model_{timestamp}"
    
    # 2. Tạo thư mục lưu
    save_path = os.path.join(save_dir, model_name)
    os.makedirs(save_path, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"💾 BẮT ĐẦU LƯU MÔ HÌNH")
    print(f"   - Thư mục lưu: {save_path}")
    print(f"{'='*60}\n")
    
    # 1. Lưu models (HMM)
    models_path = os.path.join(save_path, 'models.pkl')
    with open(models_path, 'wb') as f:
        pickle.dump(models, f)
    print(f"✅ Đã lưu models (dạng list) tại: {models_path}")
    
    # 2. Lưu scaler
    scaler_path = os.path.join(save_path, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✅ Đã lưu scaler tại: {scaler_path}")

    # 3. Chuyển đổi metrics (Numpy -> list) để lưu JSON
    metrics_serializable = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            metrics_serializable[key] = value.tolist()
        elif isinstance(value, (np.int64, np.int32, np.float64, np.float32, np.bool_)):
            metrics_serializable[key] = value.item() # Dùng .item() an toàn hơn
        else:
            metrics_serializable[key] = value
            
    metrics_path = os.path.join(save_path, 'metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        # Dùng metrics_serializable đã được xử lý
        json.dump(metrics_serializable, f, indent=4, ensure_ascii=False) 
    print(f"✅ Đã lưu metrics tại: {metrics_path}")
    
    
    # 4. Lưu thông tin tóm tắt
    summary = {
        'model_name': model_name,
        'save_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'num_classes': len(models),
        'class_names': class_names,  # Lấy từ tham số class_names
        'accuracy': float(metrics.get('accuracy', 0)),
        'f1_macro': float(metrics.get('f1_macro', 0)),
        'f1_weighted': float(metrics.get('f1_weighted', 0))
    }
    
    summary_path = os.path.join(save_path, 'summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    print(f"✅ Đã lưu summary tại: {summary_path}")
    
    # In kết quả cuối cùng
    print(f"\n{'='*60}")
    print(f"✅ HOÀN THÀNH LƯU MÔ HÌNH")
    print(f"{'='*60}")
    print(f"   📁 Thư mục: {save_path}")
    print(f"   📊 Accuracy: {summary['accuracy']:.4f}")
    print(f"   📈 F1-Score (Macro): {summary['f1_macro']:.4f}")
    print(f"{'='*60}\n")
    
    return save_path


def load_model(load_path):
    """
    Tải mô hình HMM, scaler và metrics
    
    Args:
        load_path: Đường dẫn thư mục chứa mô hình
    
    Returns:
        models: Dictionary chứa các mô hình HMM
        scaler: Scaler
        metrics: Dictionary chứa các chỉ số đánh giá
        summary: Dictionary chứa thông tin tóm tắt
    """
    print(f"\n{'='*60}")
    print(f"📂 BẮT ĐẦU TẢI MÔ HÌNH")
    print(f"{'='*60}")
    print(f"   - Đường dẫn: {load_path}")
    print(f"{'='*60}\n")
    
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

    print(f"\n{'='*60}")
    print(f"✅ HOÀN THÀNH TẢI MÔ HÌNH")
    print(f"{'='*60}")
    print(f"   📅 Ngày lưu: {summary['save_time']}")
    print(f"   🏷️  Số lớp: {summary['num_classes']}")
    print(f"   📊 Accuracy: {summary['accuracy']:.4f}")
    print(f"   📈 F1-Score (Macro): {summary['f1_macro']:.4f}")
    print(f"{'='*60}\n")
    
    return models, scaler, metrics, summary

