import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import scipy.signal
import json
import pandas as pd

yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
yamnet_model = hub.load(yamnet_model_handle)

file_path = "/content/your_wav_file.wav"

from pydub import AudioSegment

audio = AudioSegment.from_mp3("/content/snore_noise_1min (1).mp3")

audio.export("your_wav_file.wav", format="wav")

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
    38: "Snoring",
    20: "Baby cry",
    0: "Speech",
    312: "Truck horn",
    78: "Meow",
    42: "Cough",
    44: "sneeze"
}

target_indices = list(target_sounds_map.keys())
confidence_threshold = 0.85

wav_file, sample_rate = tf.audio.decode_wav(
    tf.io.read_file(file_path),
    desired_channels=1,
    desired_samples=-1,
    name=None
)

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
                'sound': detected_sound,
                'confidence': confidence,
                'time': timestamp_sec
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

# for det in filtered_detections:
#     print(f"감지됨: '{det['sound']}' (확신도: {det['confidence']:.2f})")
#     print(f"타임스탬프: {det['time']:.2f}초")

snoring_detections = [det for det in filtered_detections if det['sound'] == 'Snoring']

apnea_events = []
last_snore_time = 0.0
apnea_threshold_min = 10.0 # 최소 무호흡 시간
apnea_threshold_max = 15.0 # 최대 무호흡 시간

for i, current_detection in enumerate(snoring_detections):
    current_time = current_detection['time']

    time_gap = current_time - last_snore_time

    if apnea_threshold_min <= time_gap <= apnea_threshold_max:
        apnea_start_time = last_snore_time
        apnea_end_time = current_time

        apnea_events.append({
            'start_time': apnea_start_time,
            'end_time': apnea_end_time,
            'duration': time_gap
        })

    last_snore_time = current_time

total_audio_duration = wav_file.shape[0] / 16000.0

time_gap_at_end = total_audio_duration - last_snore_time
if apnea_threshold_min <= time_gap_at_end <= apnea_threshold_max:
    apnea_events.append({
        'start_time': last_snore_time,
        'end_time': total_audio_duration,
        'duration': time_gap_at_end
    })

for event in apnea_events:
    print(f"무호흡증 감지 : {event['start_time']:.2f}초 ~ {event['end_time']:.2f}초, 지속 시간: {event['duration']:.2f}초")
    
simplified_detections = []
for det in filtered_detections:
    simplified_detections.append({
        'sound': det['sound'],
        'time': det['time']
    })

# 새로운 딕셔너리에 담기
json_data = {
    "detected_sounds": simplified_detections
}

# JSON 파일로 저장
json_filepath = "/content/detected_sounds_simplified.json"
with open(json_filepath, "w", encoding='utf-8') as f:
    json.dump(json_data, f, indent=2)

print(f"JSON 파일 저장 완료: {json_filepath}")