from __future__ import annotations

from datetime import timedelta
from typing import List, Dict, Any
import logging

import numpy as np
import subprocess
import tensorflow as tf  # type: ignore
import tensorflow_hub as hub  # type: ignore

from app.api.domain.domain.entity.analyzed_data_entity import SoundEventType
from app.common import config
from app.api.domain.domain.vo.analyzed_data_value_object import SoundEventData
from app.api.domain.domain.vo.chunked_data_value_object import SoundChunkData


class SoundAnalyzerService:
    def __init__(self) -> None:
        self._model = None
        self._class_map: Dict[int, str] = {}
        self._snore_index: int | None = None
        # Confidence threshold aligned with notebook logic
        self._confidence_threshold = 0.7
        # Preload model and class map to avoid cold-start latency
        self._ensure_model()
        self._ensure_class_map()

    def _ensure_model(self) -> None:
        if self._model is None:
            self._model = hub.load(config.SOUND_YAMNET_MODEL_URL)

    def _ensure_class_map(self) -> None:
        if not self._class_map:
            import csv
            from app.common import config
            class_map_url = config.SOUND_YAMNET_CLASS_MAP_URL
            class_map_filename = config.SOUND_YAMNET_CLASS_MAP_FILENAME
            path = tf.keras.utils.get_file(class_map_filename, class_map_url)
            with tf.io.gfile.GFile(path, "r") as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    if len(row) >= 3:
                        idx = int(row[0])
                        name = row[2]
                        self._class_map[idx] = name
            # Precompute Snoring index by display_name match
            try:
                self._snore_index = next(i for i, n in self._class_map.items() if n.strip().lower() == "snoring".lower())
            except StopIteration:
                self._snore_index = None

    def _decode_mp3_to_float(self, data: bytes, *, sample_rate: int = 16000, channels: int = 1) -> np.ndarray:
        cmd = [
            "ffmpeg",
            "-v", "error",
            "-i", "pipe:0",
            "-ac", str(channels),
            "-ar", str(sample_rate),
            "-f", "s16le",
            "pipe:1",
        ]
        pcm = subprocess.check_output(cmd, input=data, stderr=subprocess.STDOUT, timeout=60)
        samples = np.frombuffer(pcm, dtype=np.int16)
        return samples.astype(np.float32) / 32768.0

    def _label_to_event_type(self, label: str) -> SoundEventType | None:
        # Map YAMNet label strings to this project's SoundEventType enums
        name = label.strip().lower()
        if "snor" in name:
            return SoundEventType.SNORE
        if "cough" in name:
            return SoundEventType.COUGH
        if ("baby" in name and "cry" in name) or ("infant" in name and "cry" in name):
            return SoundEventType.BABY_CRYING
        if "breath" in name:
            return SoundEventType.MOUTH_BREATHING
        if any(x in name for x in ["cat", "meow", "dog", "bark", "animal", "bird"]):
            return SoundEventType.ANIMAL_NOISE
        if "horn" in name or "car horn" in name or "truck horn" in name:
            return SoundEventType.CAR_HORN
        return None

    def analyze(self, chunks: List[SoundChunkData]) -> List[SoundEventData]:
        # Detect all supported events by mapping top-N labels per frame
        self._ensure_model()
        self._ensure_class_map()
        events: List[SoundEventData] = []
        top_n = 5
        for c in chunks:
            x = self._decode_mp3_to_float(c.data)
            x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
            scores, _, _ = self._model(x_tensor)
            frame_times = np.arange(scores.shape[0]) * 0.48
            scores_np = scores.numpy()
            for i in range(len(scores_np)):
                frame_scores = scores_np[i]
                top_indices = np.argsort(frame_scores)[-top_n:][::-1]
                for idx in top_indices:
                    conf = float(frame_scores[idx])
                    if conf < self._confidence_threshold:
                        continue
                    label = self._class_map.get(int(idx), "")
                    ev_type = self._label_to_event_type(label)
                    if ev_type is None:
                        continue
                    events.append(
                        SoundEventData(
                            event=ev_type,
                            recorded_at=c.start_at + timedelta(seconds=float(frame_times[i])),
                        )
                    )
        return events

    def analyze_chunk_to_detections(self, chunk: SoundChunkData, *, top_n: int = 5, confidence_threshold: float = 0.7) -> List[Dict[str, Any]]:
        self._ensure_model()
        self._ensure_class_map()
        x = self._decode_mp3_to_float(chunk.data)
        x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
        scores, _, _ = self._model(x_tensor)
        detections: List[Dict[str, Any]] = []
        frame_times = np.arange(scores.shape[0]) * 0.48
        scores_np = scores.numpy()
        for i in range(len(scores_np)):
            frame_scores = scores_np[i]
            top_indices = np.argsort(frame_scores)[-top_n:][::-1]
            for idx in top_indices:
                conf = float(frame_scores[idx])
                if conf < confidence_threshold:
                    continue
                label = self._class_map.get(int(idx), "Unknown")
                detections.append({
                    "sound": label,
                    "confidence": conf,
                    "time_in_chunk": float(frame_times[i]),
                })
        return detections

    def analyze_webm_bytes(self, webm_bytes: bytes, *, start_at_seconds: float = 0.0, top_n: int = 5, confidence_threshold: float = 0.7) -> List[Dict[str, Any]]:
        from datetime import datetime, timezone
        start_at = datetime.fromtimestamp(start_at_seconds, tz=timezone.utc)
        chunk = SoundChunkData(data=webm_bytes, start_at=start_at, sampling_rate_hz=16000.0)
        return self.analyze_chunk_to_detections(chunk, top_n=top_n, confidence_threshold=confidence_threshold)


