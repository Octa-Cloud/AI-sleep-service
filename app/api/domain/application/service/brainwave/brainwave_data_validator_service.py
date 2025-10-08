from __future__ import annotations

import os
import tempfile
import pyedflib
from typing import Iterable

from app.api.common.exception.custom.brainwave_exceptions import (
    BrainwaveFormatValidationFailApiException,
    BrainwaveChannelValidationFailApiException,
    BrainwaveLengthValidationFailApiException
)


class BrainwaveDataValidatorService:
    """Validate EDF bytes for required channels and duration constraints.

    Rules:
    - File must be a decodable EDF. Otherwise raise BrainwaveFormatValidationFailApiException.
    - Must contain both channels: 'Fpz-Cz' and 'Pz-Oz' (case-insensitive, allow optional 'EEG ' prefix).
      Otherwise raise BrainwaveChannelValidationFailApiException.
    - Total length must be 10 minutes ± 30 seconds (570s ~ 630s).
      Otherwise raise BrainwaveLengthValidationFailApiException.
    """

    # Case-sensitive requirement with optional 'EEG ' prefix.
    REQUIRED_CHANNELS_EXACT = {"Fpz-Cz", "Pz-Oz"}
    MIN_DURATION_SECONDS = 600 - 30
    MAX_DURATION_SECONDS = 600 + 30

    def validate(self, edf_bytes: bytes) -> None:
        if not edf_bytes or len(edf_bytes) == 0:
            raise BrainwaveFormatValidationFailApiException()

        if pyedflib is None:
            # In runtime environments without pyedflib installed correctly
            raise BrainwaveFormatValidationFailApiException()

        tmp_path = None
        try:
            # pyedflib reads from file path; persist to a temp file safely.
            with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
                tmp.write(edf_bytes)
                tmp.flush()
                tmp_path = tmp.name

            reader = pyedflib.EdfReader(tmp_path)
            try:
                labels: list[str] = list(reader.getSignalLabels())
                duration_seconds: float = float(reader.getFileDuration())
            finally:
                reader.close()

            if not labels:
                raise BrainwaveFormatValidationFailApiException()

            normalized = self._normalize_labels(labels)
            if not self.REQUIRED_CHANNELS_EXACT.issubset(normalized):
                raise BrainwaveChannelValidationFailApiException()

            if not (self.MIN_DURATION_SECONDS <= duration_seconds <= self.MAX_DURATION_SECONDS):
                raise BrainwaveLengthValidationFailApiException()

        except BrainwaveFormatValidationFailApiException:
            raise
        except BrainwaveChannelValidationFailApiException:
            raise
        except BrainwaveLengthValidationFailApiException:
            raise
        except Exception:
            # Any other decoding/parsing errors are treated as format failures per spec
            raise BrainwaveFormatValidationFailApiException()
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    @staticmethod
    def _normalize_labels(labels: Iterable[str]) -> set[str]:
        normalized: set[str] = set()
        for label in labels:
            value = label.strip()
            if value.startswith("EEG "):
                value = value[4:]
            normalized.add(value)
        return normalized


