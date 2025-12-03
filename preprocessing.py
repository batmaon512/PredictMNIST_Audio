import os
import numpy as np
import soundfile as sf
from python_speech_features import mfcc, delta

SR = 22050
N_MFCC = 13
NFFT = 2048  # lớn hơn độ dài frame để tránh cảnh báo

def extract_features(file_path):
    try:
        audio, sr = sf.read(file_path, always_2d=False)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        audio = np.asarray(audio, dtype=np.float32)

        # Nếu sample rate khác SR, vẫn dùng sr hiện có để tránh resample nặng
        # Pre-emphasis
        audio = np.append(audio[0], audio[1:] - 0.95 * audio[:-1])

        # MFCC 13 dims
        mfcc_ = mfcc(signal=audio, samplerate=sr, numcep=N_MFCC, nfft=NFFT)

        # Delta và Delta-Delta
        d1 = delta(mfcc_, 2)
        d2 = delta(d1, 2)

        # Ghép thành 39 dims theo trục đặc trưng
        feats = np.hstack([mfcc_, d1, d2])  # shape: (frames, 39)
        return feats
    except Exception as e:
        print(f"Lỗi khi xử lý file {file_path}: {e}")
        return None

def read_dataset_and_save_feture(root_folder_path, save_path):
    features_list = []
    labels_list = []
    
    if not os.path.exists(root_folder_path):
        print(f"Lỗi: Không tìm thấy thư mục '{root_folder_path}'")
        return

    for label in tqdm(os.listdir(root_folder_path), desc="Processing Labels"):
        folder_path = os.path.join(root_folder_path, label)
        if not os.path.isdir(folder_path):
            continue
            
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            features = extract_features(file_path)
            if features is not None:
                features_list.append(features)
                labels_list.append(label)

    # Lưu dưới dạng object array để chứa các chuỗi có độ dài khác nhau
    np.savez_compressed(save_path, features=np.array(features_list, dtype=object), labels=np.array(labels_list))
    print(f"\nTrích xuất hoàn tất! Đã lưu {len(features_list)} chuỗi đặc trưng.")

def load_and_preprocess_data(data_path):
    data = np.load(data_path, allow_pickle=True)
    X_list = data['features']
    y_str = data['labels']

    # Mã hóa nhãn (giữ nguyên như cũ)
    label_map = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
        'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9
    }
    class_names = sorted(label_map, key=label_map.get)
    y_encoded = np.array([label_map[label] for label in y_str])

    return X_list, y_encoded, class_names

def split_train_test(X_list, y, test_size=0.2, random_state=42):
    # Cần chuyển X_list thành mảng tạm để stratify hoạt động
    indices = np.arange(len(X_list))
    train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=random_state, stratify=y)
    
    X_train = [X_list[i] for i in train_indices]
    X_test = [X_list[i] for i in test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    X_concatenated = np.vstack(X_train)
    scaler = StandardScaler()
    scaler.fit(X_concatenated)
    X_train = [scaler.transform(x) for x in X_train]
    X_test = [scaler.transform(x) for x in X_test]
    return X_train, X_test, y_train, y_test, scaler