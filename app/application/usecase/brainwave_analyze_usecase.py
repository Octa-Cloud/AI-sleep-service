import tensorflow as tf
import numpy as np
import mne
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from collections import deque

base_dir = "C:\\AI-sleep-service\\app"

try:
    std_path = os.path.join(base_dir, "models", "std.npy")
    model_path = os.path.join(base_dir, "models", "model_4.keras")
    mean_path = os.path.join(base_dir, "models", "mean.npy")
    edf_file = os.path.join(base_dir, "psg_file", "ST7242J0-PSG.edf")
except FileNotFoundError as e:
    print(f"오류: 필요한 파일을 찾을 수 없습니다. 경로를 확인해주세요. {e}")
    exit()

def preprocess_segment(segment_data, samples_per_epoch):
    n_epochs = segment_data.shape[1] // samples_per_epoch
    epochs = np.array(np.split(segment_data[:, :n_epochs * samples_per_epoch], n_epochs, axis=1))
    return np.transpose(epochs, (0, 2, 1)).astype(np.float32)

def analyze_segment_batch(segment_data_batch, mean, std, model):
    data_for_prediction = np.concatenate(segment_data_batch, axis=0)
    data_for_prediction = (data_for_prediction - mean) / std
    predictions = model.predict(data_for_prediction, verbose=0)
    return np.argmax(predictions, axis=1)

if __name__ == '__main__':
    try:
        model = tf.keras.models.load_model(model_path)
        mean = np.load(mean_path).astype(np.float32)
        std = np.load(std_path).astype(np.float32)
        print("모델과 정규화 파일 로드 완료")
    except Exception as e:
        print(f"모델 또는 정규화 파일 로드 중 오류 발생: {e}")
        exit()

    raw = mne.io.read_raw_edf(edf_file, preload=True)
    raw.pick_channels(["EEG Fpz-Cz", "EEG Pz-Oz"])
    raw.filter(l_freq=0.5, h_freq=30.0, fir_design='firwin')

    sfreq = raw.info['sfreq']
    epoch_duration = 30
    segment_duration = 15 * 60
    
    samples_per_second = int(sfreq)
    samples_per_segment = int(segment_duration * sfreq)
    samples_per_epoch = int(epoch_duration * sfreq)

    total_samples = raw.n_times
    total_duration_minutes = total_samples / sfreq / 60

    data_buffer = deque(maxlen=samples_per_segment)
    all_predicted_classes = []

    print(f"총 {total_duration_minutes:.2f}분 분량의 뇌파 데이터를 시뮬레이션합니다.")

    start_time_sim = time.time()
    for i in range(0, total_samples, samples_per_second):
        chunk_end = min(i + samples_per_second, total_samples)
        chunk = raw.get_data(start=i, stop=chunk_end)
        
        if chunk.size > 0:
            for j in range(chunk.shape[1]):
                data_buffer.append(chunk[:, j])
        
        if len(data_buffer) >= samples_per_segment:
            print(f"현재까지 {i/sfreq/60:.2f}분 경과. 15분 세그먼트 분석을 시작합니다.")
            
            segment_data = np.array(list(data_buffer)).T
            data_buffer.clear()
            
            with ProcessPoolExecutor() as executor:
                future = executor.submit(preprocess_segment, segment_data, samples_per_epoch)
                preprocessed_segment = future.result()
                
                predicted_classes = analyze_segment_batch([preprocessed_segment], mean, std, model).tolist()
                all_predicted_classes.extend(predicted_classes)

            print(f"15분 세그먼트 분석 완료. 현재까지 총 {len(all_predicted_classes)} 에포크 예측.")

            for j in range(1, len(all_predicted_classes) - 1):
                if (all_predicted_classes[j - 1] == 5 and
                    all_predicted_classes[j] == 2 and
                    all_predicted_classes[j + 1] == 5):
                    print(f"인덱스 {j}: 5-2-5 패턴 발견! 값 2를 5로 변경합니다.")
                    all_predicted_classes[j] = 5
        
    print("--- 모든 데이터 스트리밍 시뮬레이션 완료 ---")
    
    is_zero = (np.array(all_predicted_classes) == 0).astype(int)
    if len(is_zero) > 30:
        zero_run = np.max(np.convolve(is_zero, np.ones(30, dtype=int), mode='valid'))
        if zero_run >= 30:
            first_zero_index = np.where(is_zero.astype(bool))[0][0]
            all_predicted_classes = all_predicted_classes[:first_zero_index + 30]

    json_data = {
        "file": os.path.basename(edf_file),
        "sampling_rate": sfreq,
        "total_epochs": len(all_predicted_classes),
        "predicted_classes": all_predicted_classes
    }

    base_filename = os.path.splitext(os.path.basename(edf_file))[0]
    json_filename = f"{base_filename}.predict.json"
    json_filepath = os.path.join(base_dir, "psg_file", json_filename)

    with open(json_filepath, "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"JSON 파일 저장 완료: {json_filepath}")