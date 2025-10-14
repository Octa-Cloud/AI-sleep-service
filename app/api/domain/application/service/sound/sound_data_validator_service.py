from __future__ import annotations

import subprocess
import json

from app.api.common.exception.custom.sound_exceptions import (
    SoundFormatValidationFailApiException,
    SoundChannelValidationFailApiException,
    SoundLengthValidationFailApiException,
    SoundSampleRateValidationFailApiException,
)
from app.api.common.exception.api_exception import ApiException


class SoundDataValidatorService:
    def validate(self, data: bytes) -> None:
        # Stream input via stdin to avoid temp files
        try:
            # Probe stream and container
            cmd = [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=format_name,duration:stream=index,codec_name,channels,sample_rate",
                "-of", "json",
                "-i", "pipe:0",
            ]
            raw = subprocess.check_output(cmd, input=data, stderr=subprocess.STDOUT, timeout=10).decode()
            info = json.loads(raw)
            fmt = (info.get("format") or {}).get("format_name", "")
            dur_s = float((info.get("format") or {}).get("duration", "0") or 0)
            streams = info.get("streams") or []
            a0 = next((s for s in streams if s.get("index", 0) == 0 or s.get("codec_type") == "audio"), None)
            codec = (a0 or {}).get("codec_name", "")
            channels = int((a0 or {}).get("channels", 0) or 0)
            sr = int((a0 or {}).get("sample_rate", 0) or 0)

            if ("webm" not in fmt and "matroska" not in fmt) or codec != "opus":
                raise SoundFormatValidationFailApiException()
            if dur_s > 60.5:
                raise SoundLengthValidationFailApiException()
            # Optionally enforce mono/16k when required
            # if channels != 1:
            #     raise SoundChannelValidationFailApiException()
            # if sr != 16000:
            #     raise SoundSampleRateValidationFailApiException()
        except Exception as e:
            print(f"error: {e}")
            if isinstance(e, ApiException):
                raise
            raise SoundFormatValidationFailApiException() from e


