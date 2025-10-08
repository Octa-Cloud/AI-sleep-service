from __future__ import annotations

import os
from typing import List, Iterable
from datetime import timedelta

import numpy as np
import tensorflow as tf
from concurrent.futures import ProcessPoolExecutor

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
            # if n_epochs <= 0:
            #     return np.empty((0, 0, 0), dtype=np.float32)
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

        # Robust normalization: prefer provided mean/std if time length matches; otherwise per-epoch z-score per channel
        def _normalize(arr: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
            B, T, C = arr.shape
            use_given = False
            m = mean
            s = std
            try:
                if m.ndim == 3 and m.shape == (1, T, C) and s.ndim == 3 and s.shape == (1, T, C):
                    use_given = True
                elif m.ndim == 1 and m.shape[0] == T and C == 1:
                    m = m.reshape(1, T, 1)
                    s = s.reshape(1, T, 1)
                    use_given = True
                elif m.ndim == 1 and m.shape[0] == T and C == 2:
                    m = np.stack([m, m], axis=-1).reshape(1, T, 2)
                    s = np.stack([s, s], axis=-1).reshape(1, T, 2)
                    use_given = True
            except Exception:
                use_given = False
            if use_given:
                return (arr - m) / np.where(s == 0, 1.0, s)
            # Fallback: per-epoch per-channel z-score across time
            mean_tc = np.mean(arr, axis=1, keepdims=True)  # (B,1,C)
            std_tc = np.std(arr, axis=1, keepdims=True)
            std_tc = np.where(std_tc == 0, 1.0, std_tc)
            return (arr - mean_tc) / std_tc

        data_for_prediction = _normalize(data_for_prediction, self._mean, self._std)
        # Use process pool to parallelize prediction by chunks to avoid GIL contention
        def _process_segment_batch(args):
            segment_batch, mean, std, model_path = args
            # Load model in worker to avoid TensorFlow graph clashes across processes
            model = tf.keras.models.load_model(model_path)
            arr = np.concatenate(segment_batch, axis=0)
            # Apply same normalization rule in worker
            B, T, C = arr.shape
            use_given = False
            m = mean
            s = std
            try:
                if m.ndim == 3 and m.shape == (1, T, C) and s.ndim == 3 and s.shape == (1, T, C):
                    use_given = True
                elif m.ndim == 1 and m.shape[0] == T and C == 1:
                    m = m.reshape(1, T, 1)
                    s = s.reshape(1, T, 1)
                    use_given = True
                elif m.ndim == 1 and m.shape[0] == T and C == 2:
                    m = np.stack([m, m], axis=-1).reshape(1, T, 2)
                    s = np.stack([s, s], axis=-1).reshape(1, T, 2)
                    use_given = True
            except Exception:
                use_given = False
            if use_given:
                arr = (arr - m) / np.where(s == 0, 1.0, s)
            else:
                mean_tc = np.mean(arr, axis=1, keepdims=True)
                std_tc = np.std(arr, axis=1, keepdims=True)
                std_tc = np.where(std_tc == 0, 1.0, std_tc)
                arr = (arr - mean_tc) / std_tc
            preds = model.predict(arr, verbose=0)
            return np.argmax(preds, axis=1)

        # Build per-segment batches for workers
        segment_batches: List[np.ndarray] = batch_list
        if len(segment_batches) == 1:
            predictions = self._model.predict((data_for_prediction), verbose=0)
            classes = np.argmax(predictions, axis=1).astype(int).tolist()
        else:
            tasks = [( [seg], self._mean, self._std, self._model_path ) for seg in segment_batches]
            classes_arrays: List[np.ndarray] = []
            with ProcessPoolExecutor(max_workers=int(os.getenv("BRAINWAVE_PROC_WORKERS", "2"))) as executor:
                for result in executor.map(_process_segment_batch, tasks):
                    classes_arrays.append(result.astype(int))
            classes = np.concatenate(classes_arrays, axis=0).astype(int).tolist()

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


