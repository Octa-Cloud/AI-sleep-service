from __future__ import annotations

import os
import tempfile
from typing import List
from datetime import datetime, timedelta
import pyedflib
import numpy as np
try:
    from scipy.signal import butter, filtfilt, firwin  # type: ignore
except Exception:
    butter = filtfilt = firwin = None  # type: ignore

from app.api.common.exception.custom.brainwave_exceptions import (
    BrainwaveFormatValidationFailApiException,
)
from app.api.domain.domain.vo.chunked_data_value_object import BrainwaveChunkData


class BrainwaveChunkSplitterService:
    def split(self, edf_bytes: bytes) -> List[BrainwaveChunkData]:
        if pyedflib is None:
            raise BrainwaveFormatValidationFailApiException()

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
                tmp.write(edf_bytes)
                tmp.flush()
                tmp_path = tmp.name

            reader = pyedflib.EdfReader(tmp_path)
            try:
                labels = list(reader.getSignalLabels())
                # EDF start datetime (UTC naive)
                startdate = reader.getStartdatetime()  # datetime
                # Pick exact case-sensitive channels, with optional 'EEG ' prefix
                def pick(label: str) -> int:
                    arr = reader.getSignalLabels()
                    for idx, lab in enumerate(arr):
                        name = lab.strip()
                        if name.startswith("EEG "):
                            name = name[4:]
                        if name == label:
                            return idx
                    raise BrainwaveFormatValidationFailApiException()

                idx_fpz = pick("Fpz-Cz")
                idx_pz = pick("Pz-Oz")

                sfreq = float(reader.getSampleFrequency(idx_fpz))
                n_times = int(reader.getNSamples()[idx_fpz])

                data_fpz = reader.readSignal(idx_fpz)
                data_pz = reader.readSignal(idx_pz)
                data = np.vstack([data_fpz, data_pz]).astype(np.float32)  # shape: (2, n_times)

                # Optional band-pass filter 0.5–30 Hz (default butter; set BRAINWAVE_FILTER_IMPL=fir to use firwin)
                if filtfilt is not None:
                    impl = os.getenv("BRAINWAVE_FILTER_IMPL", "butter").lower()
                    low, high = 0.5, 30.0
                    nyq = 0.5 * sfreq
                    if impl == "fir" and firwin is not None:
                        taps = firwin(numtaps=513, cutoff=[low / nyq, high / nyq], pass_zero=False)
                        for ch in range(data.shape[0]):
                            data[ch, :] = filtfilt(taps, [1.0], data[ch, :])
                    elif butter is not None:
                        b, a = butter(4, [low / nyq, high / nyq], btype="bandpass")
                        for ch in range(data.shape[0]):
                            data[ch, :] = filtfilt(b, a, data[ch, :], method="gust")

            finally:
                reader.close()

            epoch_sec = 30
            # Option to emit whole file as a single chunk (for parity with offline script)
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


