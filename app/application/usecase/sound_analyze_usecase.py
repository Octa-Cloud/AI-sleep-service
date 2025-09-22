import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import scipy.signal
import json
import os

# YAMNet 모델 로드
yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
yamnet_model = hub.load(yamnet_model_handle)

# 분석할 파일 경로 설정 (WAV 파일로 직접 지정)
file_path = "C:\\AI-sleep-service\\app\\sound_file\\sound_data.wav"

from pydub import AudioSegment
try:
    audio = AudioSegment.from_wav(file_path)
    audio = audio.set_sample_width(2) # 2바이트 = 16비트
    audio.export(file_path, format="wav") # 원본 파일 덮어쓰기
except Exception as e:
    print(f"Error converting file to 16-bit: {e}")

class_map_path = yamnet_model.class_map_path().numpy()
class_map = {}
with tf.io.gfile.GFile(class_map_path, 'r') as f:
    _ = f.readline()
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 3:
            index = int(parts[0])
            name = parts[2]
            class_map[index] = name

target_sounds_map = {
    38: "Snoring", 20: "Baby cry", 0: "Speech",
    312: "Truck horn", 78: "Meow", 42: "Cough"
}

target_indices = list(target_sounds_map.keys())
confidence_threshold = 0.9

wav_file, sample_rate = tf.audio.decode_wav(
    tf.io.read_file(file_path),
    desired_channels=1,
    desired_samples=-1,
    name=None
)

# 데이터 타입을 float32로 변환
wav_file = tf.cast(wav_file, dtype=tf.float32)

if tf.constant(sample_rate) != 16000:
    wav_numpy = wav_file.numpy().flatten()
    num_samples = int(len(wav_numpy) * 16000 / sample_rate.numpy())
    wav_resampled = scipy.signal.resample(wav_numpy, num_samples)
    wav_file = tf.constant(wav_resampled, dtype=tf.float32)

scores, embeddings, spectrogram = yamnet_model(wav_file)
yamnet_frames = scores.shape[0]
timestamps = np.arange(yamnet_frames) * 0.48
detections = []
frame_times = np.arange(scores.shape[0]) * 0.48

for i in range(len(scores)):
    frame_scores = scores[i].numpy()
    for idx in target_indices:
        confidence = frame_scores[idx]
        if confidence > confidence_threshold:
            detected_sound = target_sounds_map[idx]
            timestamp_sec = frame_times[i]
            detections.append({
                'sound': detected_sound, 'confidence': confidence, 'time': timestamp_sec
            })

filtered_detections = []
last_detected_time = -1.0
detection_interval = 1.0
detections.sort(key=lambda x: x['time'])

for detection in detections:
    current_time = detection['time']
    if current_time - last_detected_time >= detection_interval:
        filtered_detections.append(detection)
        last_detected_time = current_time

for det in filtered_detections:
    print(f"감지됨: '{det['sound']}' (확신도: {det['confidence']:.2f})")
    print(f"타임스탬프: {det['time']:.2f}초")

simplified_detections = []
for det in filtered_detections:
    simplified_detections.append({
        'sound': det['sound'], 'time': det['time']
    })

json_data = {"detected_sounds": simplified_detections}
json_filepath = "C:\\AI-sleep-service\\app\\prediction\\sounds_simplified.json"

output_dir = os.path.dirname(json_filepath)

os.makedirs(output_dir, exist_ok=True)

with open(json_filepath, "w", encoding='utf-8') as f:
    json.dump(json_data, f, indent=2)

print(f"JSON 파일 저장 완료: {json_filepath}")