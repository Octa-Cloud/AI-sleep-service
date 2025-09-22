import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import json
import os
from pydub import AudioSegment

YAMNET_MODEL = None
TARGET_SOUNDS_MAP = {
    38: "Snoring", 20: "Baby cry", 0: "Speech",
    312: "Truck horn", 78: "Meow", 42: "Cough"
}
CONFIDENCE_THRESHOLD = 0.87

def initialize_model():
    global YAMNET_MODEL
    if YAMNET_MODEL is None:
        try:
            print("YAMNet 모델을 로드합니다...")
            yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
            YAMNET_MODEL = hub.load(yamnet_model_handle)
            print("모델 로드 완료.")
        except Exception as e:
            print(f"치명적 오류: YAMNet 모델 로드 실패: {e}")
            raise

def analyze_audio_chunk(chunk_tensor, time_offset_sec):
    scores, _, _ = YAMNET_MODEL(chunk_tensor)
    
    detections = []
    frame_times = np.arange(scores.shape[0]) * 0.48 + time_offset_sec

    for i, frame_scores in enumerate(scores):
        for idx, sound_name in TARGET_SOUNDS_MAP.items():
            if frame_scores[idx] > CONFIDENCE_THRESHOLD:
                detections.append({
                    'sound': sound_name,
                    'time': frame_times[i]
                })
    return detections

def append_results_to_json(new_detections, output_path):
    try:
        # 1. 기존 데이터가 있으면 불러오고, 없으면 새로 생성
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            with open(output_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = {"detected_sounds": []}
        
        # 2. 기존 결과에 새로운 탐지 결과를 합침
        all_detections = data["detected_sounds"]
        all_detections.extend(new_detections)
        
        # 3. 모든 결과를 시간순으로 정렬
        all_detections.sort(key=lambda x: x['time'])

        # 4. 2초 간격 필터링 로직 적용
        filtered_detections = []
        last_accepted_time = -2.0  
        for det in all_detections:
            if det['time'] - last_accepted_time >= 2.0:
                filtered_detections.append(det)
                last_accepted_time = det['time']
        
        data["detected_sounds"] = filtered_detections
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
    except Exception as e:
        print(f"JSON 파일 저장 중 오류 발생: {e}")


def process_file_like_stream(input_wav_path, output_json_path, chunk_minutes=1):
    initialize_model()

    try:
        print(f"오디오 파일 로드 중: {input_wav_path}")
        audio = AudioSegment.from_wav(input_wav_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        print("오디오 파일 로드 완료.")
    except Exception as e:
        print(f"오디오 파일 로드 중 오류 발생: {e}")
        return
        
    chunk_length_ms = chunk_minutes * 60 * 1000
    
    
    for i, start_ms in enumerate(range(0, len(audio), chunk_length_ms)):
        end_ms = start_ms + chunk_length_ms
        chunk = audio[start_ms:end_ms]

        print(f"\n{start_ms / 1000:.1f}초 지점: 청크 #{i+1} 분석 중...")
        
        samples_int = np.array(chunk.get_array_of_samples())
        normalized_samples = samples_int.astype(np.float32) / np.iinfo(samples_int.dtype).max
        chunk_tensor = tf.convert_to_tensor(normalized_samples, dtype=tf.float32)
        
        time_offset_sec = start_ms / 1000
        detections_in_chunk = analyze_audio_chunk(chunk_tensor, time_offset_sec)

        if detections_in_chunk:
            print(f"{len(detections_in_chunk)}개의 소리 탐지. JSON 파일에 추가 및 필터링합니다.")
            append_results_to_json(detections_in_chunk, output_json_path)
        else:
            print("탐지된 소리가 없습니다.")

    print("\n\n모든 파일 처리가 완료되었습니다.")

if __name__ == "__main__":
    input_wav_path = "C:\\AI-sleep-service\\app\\prediction\\761638__naturenotesuk__breathing-and-snoring.wav"
    output_json_path = "C:\\AI-sleep-service\\app\\prediction\\realtime_analysis_results.json"
    
    if os.path.exists(output_json_path):
        os.remove(output_json_path)
        print(f"기존 결과 파일 ({output_json_path})을 삭제했습니다.")
        
    # 만약 만약에 자신의 컴퓨터가 구리다 하는 분들은 여기 옆에 chunk_minutes 라는 곳에 숫자를 
    # 1말고 2 3 4로 바꿔서 코어별로 속도 측정해서 톡으로 send 해주시면 thank you 하겠습니다.
    process_file_like_stream(input_wav_path, output_json_path, chunk_minutes=1)