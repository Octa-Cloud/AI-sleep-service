from __future__ import annotations

import os
from typing import List
from datetime import timedelta

import numpy as np
import tensorflow as tf

from app.api.domain.domain.vo.chunked_data_value_object import BrainwaveChunkData
from app.api.domain.domain.vo.analyzed_data_value_object import SleepLevelData


class BrainwaveAnalyzerService:
    def __init__(self, app_root: str | None = None) -> None:
        models_dir = os.getenv("MODELS_DIR")
        if not models_dir:
            base = app_root or os.getcwd()
            models_dir = os.path.join(base, "app", "models")

        self._model_path = os.path.join(models_dir, "model_4.keras")
        self._mean_path = os.path.join(models_dir, "mean.npy")
        self._std_path = os.path.join(models_dir, "std.npy")

        self._model = tf.keras.models.load_model(self._model_path)
        self._mean = np.load(self._mean_path).astype(np.float32)
        self._std = np.load(self._std_path).astype(np.float32)

    def analyze(self, chunks: List[BrainwaveChunkData]) -> List[SleepLevelData]:
        # Prepare segments
        segment_arrays: List[np.ndarray] = [c.data for c in chunks]
        if not segment_arrays:
            return []

        # Epoch config
        epoch_sec = 30

        def preprocess_segment(segment_data: np.ndarray, samples_per_epoch: int) -> np.ndarray:
            n_epochs = segment_data.shape[1] // samples_per_epoch
            if n_epochs <= 0:
                return np.empty((0, 0, 0), dtype=np.float32)
            epochs = np.array(np.split(segment_data[:, : n_epochs * samples_per_epoch], n_epochs, axis=1))
            return np.transpose(epochs, (0, 2, 1)).astype(np.float32)

        # Build batch per segment with segment-specific epoch size (fs may vary)
        batch_list: List[np.ndarray] = []
        epoch_counts: List[int] = []
        for c in chunks:
            samples_per_epoch = int(round(epoch_sec * float(c.sampling_rate_hz)))
            pre = preprocess_segment(c.data, samples_per_epoch)
            epoch_counts.append(0 if pre.size == 0 else pre.shape[0])
            if pre.size:
                batch_list.append(pre)

        if not batch_list:
            return []

        data_for_prediction = np.concatenate(batch_list, axis=0)
        data_for_prediction = (data_for_prediction - self._mean) / self._std
        predictions = self._model.predict(data_for_prediction, verbose=0)
        classes = np.argmax(predictions, axis=1).astype(int).tolist()

        # optional smoothing
        if len(classes) >= 3:
            for j in range(1, len(classes) - 1):
                if classes[j - 1] == 5 and classes[j] == 2 and classes[j + 1] == 5:
                    classes[j] = 5

        # Map back to VOs by segment and epoch
        vo_list: List[SleepLevelData] = []
        class_idx = 0
        for c, n_epochs in zip(chunks, epoch_counts):
            for e in range(n_epochs):
                if class_idx >= len(classes):
                    break
                recorded_at = c.start_at + timedelta(seconds=epoch_sec * e)
                vo_list.append(SleepLevelData(level=int(classes[class_idx]), recorded_at=recorded_at))
                class_idx += 1

        return vo_list


