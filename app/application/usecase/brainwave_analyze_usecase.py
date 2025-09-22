import tensorflow as tf
import numpy as np
import mne
import json
import os
from concurrent.futures import ProcessPoolExecutor

def process_segment_worker(args):
    segment_data, samples_per_epoch, mean, std, model_path = args
    
    model = tf.keras.models.load_model(model_path)
    
    preprocessed_segment = preprocess_segment(segment_data, samples_per_epoch)
    predicted_classes = analyze_segment_batch([preprocessed_segment], mean, std, model)
    
    return predicted_classes.tolist()

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
    # --- 파일 경로 설정 ---
    base_dir = "C:\\AI-sleep-service\\app"
    try:
        std_path = os.path.join(base_dir, "models", "std.npy")
        model_path = os.path.join(base_dir, "models", "model_4.keras")
        mean_path = os.path.join(base_dir, "models", "mean.npy")
        edf_file = os.path.join(base_dir, "psg_file", "ST7242J0-PSG.edf")
        mean = np.load(mean_path).astype(np.float32)
        std = np.load(std_path).astype(np.float32)
    except Exception as e:
        print(f"필요 파일 로드 중 오류 발생: {e}")
        exit()

    raw = mne.io.read_raw_edf(edf_file, preload=True)
    raw.pick_channels(["EEG Fpz-Cz", "EEG Pz-Oz"])
    raw.filter(l_freq=0.5, h_freq=30.0, fir_design='firwin')

    sfreq = raw.info['sfreq']
    epoch_duration = 30
    segment_duration = 10 * 60
    samples_per_segment = int(segment_duration * sfreq)
    samples_per_epoch = int(epoch_duration * sfreq)
    total_samples = raw.n_times

    tasks = []
    for i in range(0, total_samples, samples_per_segment):
        chunk_end = min(i + samples_per_segment, total_samples)
        if (chunk_end - i) < samples_per_segment:
            continue
        
        segment_data = raw.get_data(start=i, stop=chunk_end)
        tasks.append((segment_data, samples_per_epoch, mean, std, model_path))

    # 여기 코어 숫자 바꿔가면서 cpu 결과 확인해서 톡으로 send 해주시면 thank you 하겠습니다.
    # cpu 성능은 mac은 모르겠지만 window는 ctrl+shift+esc 누르면 나옵니다.
    NUM_CORES = 2

    all_predicted_classes = []
    with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
        results = executor.map(process_segment_worker, tasks)
        
        for predicted_classes in results:
            all_predicted_classes.extend(predicted_classes)

    for j in range(1, len(all_predicted_classes) - 1):
        if (all_predicted_classes[j - 1] == 5 and
            all_predicted_classes[j] == 2 and
            all_predicted_classes[j + 1] == 5):
            all_predicted_classes[j] = 5
    
    is_zero = (np.array(all_predicted_classes) == 0).astype(int)
    if len(is_zero) > 30:
        zero_run = np.max(np.convolve(is_zero, np.ones(30, dtype=int), mode='valid'))
        if zero_run >= 30:
            first_zero_index = np.where(is_zero.astype(bool))[0][0]
            all_predicted_classes = all_predicted_classes[:first_zero_index + 30]

    json_data = {
        "file": os.path.basename(edf_file),
        "sampling_rate": sfreq,
        "predicted_classes": all_predicted_classes
    }

    base_filename = os.path.splitext(os.path.basename(edf_file))[0]
    json_filename = f"{base_filename}.predict.json"
    json_filepath = os.path.join(base_dir, "psg_file", json_filename)

    with open(json_filepath, "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"JSON 파일 저장 완료: {json_filepath}")
