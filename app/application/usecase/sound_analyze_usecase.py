import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import json
import os
from pydub import AudioSegment
import io # 메모리 상의 바이트 데이터를 파일처럼 다루기 위해 추가

YAMNET_MODEL = None
TARGET_SOUNDS_MAP = {
    38: "Snoring", 20: "Baby cry", 0: "Speech",
    312: "Truck horn", 78: "Meow", 42: "Cough"
}
CONFIDENCE_THRESHOLD = 0.87

# 모델 초기화 함수 (기존과 동일)
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

# 오디오 청크 분석 함수 (기존과 동일)
def analyze_audio_chunk(chunk_tensor, time_offset_sec):
    scores, _, _ = YAMNET_MODEL(chunk_tensor)
    
    detections = []
    # YAMNet은 0.96초 윈도우를 0.48초씩 겹치며 분석합니다.
    frame_times = np.arange(scores.shape[0]) * 0.48 + time_offset_sec

    for i, frame_scores in enumerate(scores):
        for idx, sound_name in TARGET_SOUNDS_MAP.items():
            if frame_scores[idx] > CONFIDENCE_THRESHOLD:
                detections.append({
                    'sound': sound_name,
                    'time': frame_times[i]
                })
    return detections

# JSON 결과 저장 함수 (기존과 동일)
def append_results_to_json(new_detections, output_path):
    try:
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            with open(output_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = {"detected_sounds": []}
        
        all_detections = data["detected_sounds"]
        all_detections.extend(new_detections)
        all_detections.sort(key=lambda x: x['time'])

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

# === 핵심 변경 사항 ===
# 기존: 파일 경로를 받아서 처리
# 변경: 압축된 오디오 '바이트 데이터'를 직접 받아서 처리하는 함수
def process_compressed_audio_bytes(audio_bytes, original_format, output_json_path):
    """
    압축된 오디오 바이트 스트림을 받아 압축 해제하고 분석합니다.
    (Kafka 컨슈머가 이 함수를 호출한다고 생각하면 됩니다)

    :param audio_bytes: Kafka 등에서 받은 오디오 데이터 (e.g., Opus, MP3)
    :param original_format: 원본 데이터 포맷 ('opus', 'mp3', 'webm' 등)
    :param output_json_path: 분석 결과를 저장할 JSON 파일 경로
    """
    initialize_model()

    try:
        print(f"수신된 {original_format} 데이터(크기: {len(audio_bytes) / 1024:.1f} KB)를 메모리에서 로드합니다...")
        # 1. 바이트 데이터를 메모리 상의 파일처럼 만듭니다.
        audio_stream = io.BytesIO(audio_bytes)
        
        # 2. pydub으로 포맷에 맞춰 오디오를 로드합니다. (압축 해제 발생)
        audio = AudioSegment.from_file(audio_stream, format=original_format)
        
        # 3. YAMNet을 위해 16kHz, 모노로 변환합니다.
        audio = audio.set_frame_rate(16000).set_channels(1)
        print("오디오 데이터 압축 해제 및 전처리 완료.")

    except Exception as e:
        print(f"메모리 내 오디오 데이터 처리 중 오류 발생: {e}")
        return
        
    # --- 이하 분석 로직은 기존과 거의 동일 ---
    print(f"\n오디오(길이: {len(audio) / 1000:.1f}초) 분석 중...")
    
    samples_int = np.array(audio.get_array_of_samples())
    # 정규화: int16 -> float32 (-1.0 ~ 1.0)
    normalized_samples = samples_int.astype(np.float32) / np.iinfo(samples_int.dtype).max
    audio_tensor = tf.convert_to_tensor(normalized_samples, dtype=tf.float32)
    
    detections = analyze_audio_chunk(audio_tensor, time_offset_sec=0)

    if detections:
        print(f"{len(detections)}개의 소리 탐지. JSON 파일에 추가 및 필터링합니다.")
        append_results_to_json(detections, output_json_path)
    else:
        print("탐지된 소리가 없습니다.")

    print("\n\n분석이 완료되었습니다.")


if __name__ == "__main__":
    # --- 시뮬레이션 환경 ---
    # 실제로는 Kafka 컨슈머가 audio_bytes와 format을 제공합니다.
    # 여기서는 로컬 파일을 Opus로 압축해서 그 과정을 흉내 내 봅니다.
    
    input_wav_path = "C:\\AI-sleep-service\\app\\prediction\\761638__naturenotesuk__breathing-and-snoring.wav"
    output_json_path = "C:\\AI-sleep-service\\app\\prediction\\realtime_analysis_results.json"
    
    if os.path.exists(output_json_path):
        os.remove(output_json_path)
        print(f"기존 결과 파일 ({output_json_path})을 삭제했습니다.")
        
    # --- 시뮬레이션 시작 ---
    print("--- Opus 압축 및 전송 시뮬레이션 시작 ---")
    try:
        # 1. 디바이스(클라이언트) 측: 원본 WAV 파일을 로드합니다.
        original_audio = AudioSegment.from_wav(input_wav_path)
        
        # 2. 전송 전 1분 길이로 자릅니다. (시연을 위해 1분만 사용)
        one_minute_chunk = original_audio[:60 * 1000] # 1분 = 60,000ms

        # 3. Opus 포맷(64kbps)으로 압축합니다. (메모리 상에서 처리)
        opus_buffer = io.BytesIO()
        one_minute_chunk.export(opus_buffer, format="opus", bitrate="64k")
        compressed_opus_bytes = opus_buffer.getvalue() # 이 데이터가 Kafka로 전송될 데이터입니다.

        print(f"원본 WAV 1분 크기: {len(one_minute_chunk.raw_data) / 1024:.1f} KB")
        print(f"압축된 Opus 1분 크기: {len(compressed_opus_bytes) / 1024:.1f} KB")
        print("-------------------------------------------\n")

        # 4. 서버(컨슈머) 측: 압축된 데이터를 받아 분석 함수를 호출합니다.
        process_compressed_audio_bytes(
            audio_bytes=compressed_opus_bytes,
            original_format="opus", # 어떤 포맷으로 받았는지 알려줌
            output_json_path=output_json_path
        )

    except Exception as e:
        print(f"시뮬레이션 중 오류 발생: {e}")
        print("pydub이 opus를 처리하려면 FFmpeg이 설치되어 있어야 합니다.")
