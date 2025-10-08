from __future__ import annotations

import os
import tempfile
from typing import List
from datetime import datetime, timedelta
import numpy as np
try:
    import mne  # type: ignore
except Exception:
    mne = None  # type: ignore

from app.api.common.exception.custom.brainwave_exceptions import (
    BrainwaveFormatValidationFailApiException,
)
from app.api.domain.domain.vo.chunked_data_value_object import BrainwaveChunkData


class BrainwaveChunkSplitterService:
    def split(self, edf_bytes: bytes) -> List[BrainwaveChunkData]:
        if mne is None:
            raise BrainwaveFormatValidationFailApiException()

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
                tmp.write(edf_bytes)
                tmp.flush()
                tmp_path = tmp.name

            # Read EDF using MNE (to match offline script pipeline)
            raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)
            # Pick channels and apply band-pass filter 0.5–30 Hz (FIR)
            try:
                raw.pick_channels(["EEG Fpz-Cz", "EEG Pz-Oz"])  # matches script labels
            except Exception:
                # Fallback: try without EEG prefix
                raw.pick_channels(["Fpz-Cz", "Pz-Oz"])  # type: ignore[arg-type]
            raw.filter(l_freq=0.5, h_freq=30.0, fir_design='firwin', verbose=False)

            sfreq = float(raw.info.get('sfreq') or 0.0)
            if sfreq <= 0:
                raise BrainwaveFormatValidationFailApiException()
            data = raw.get_data().astype(np.float32)  # (2, n_samples)
            startdate = raw.info.get('meas_date')
            if not isinstance(startdate, datetime):
                # best-effort: treat as epoch 0
                startdate = datetime.utcnow()

            epoch_sec = 30
            # Option: emit whole file as a single chunk (parity with offline)
            whole_file = os.getenv("BRAINWAVE_SPLIT_WHOLE_FILE", "0") == "1"
            segment_sec = (data.shape[1] / sfreq) if whole_file else (10 * 60)
            samples_per_segment = int(segment_sec * sfreq)

            chunks: List[BrainwaveChunkData] = []
            for start in range(0, data.shape[1], samples_per_segment):
                end = min(start + samples_per_segment, data.shape[1])
                if (end - start) < samples_per_segment:
                    continue
                segment = data[:, start:end].astype(np.float32)
                # compute segment start time using sampling rate
                segment_offset_sec = start / sfreq
                segment_start = startdate + timedelta(seconds=segment_offset_sec)
                chunks.append(BrainwaveChunkData(data=segment, start_at=segment_start, sampling_rate_hz=sfreq))

            return chunks
        except Exception:
            raise BrainwaveFormatValidationFailApiException()
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass


