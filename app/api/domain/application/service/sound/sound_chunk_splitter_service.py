from __future__ import annotations

import subprocess
from datetime import datetime, timedelta, timezone
from typing import List

import numpy as np

from app.api.domain.domain.vo.chunked_data_value_object import SoundChunkData


class SoundChunkSplitterService:
    def _decode_to_pcm_s16le(self, data: bytes, *, sr: int = 16000, channels: int = 1) -> bytes:
        cmd = [
            "ffmpeg",
            "-v", "error",
            "-i", "pipe:0",
            "-ac", str(channels),
            "-ar", str(sr),
            "-f", "s16le",
            "pipe:1",
        ]
        return subprocess.check_output(cmd, input=data, stderr=subprocess.STDOUT, timeout=60)

    def _encode_mp3(self, pcm_s16le: bytes, *, sr: int, channels: int = 1, bitrate: str = "32k") -> bytes:
        cmd = [
            "ffmpeg",
            "-v", "error",
            "-f", "s16le",
            "-ar", str(sr),
            "-ac", str(channels),
            "-i", "pipe:0",
            "-c:a", "libmp3lame",
            "-b:a", bitrate,
            "-f", "mp3",
            "pipe:1",
        ]
        return subprocess.check_output(cmd, input=pcm_s16le, stderr=subprocess.STDOUT, timeout=60)

    def split(self, sound_bytes: bytes) -> List[SoundChunkData]:
        sr = 16000
        channels = 1
        start = datetime.now(timezone.utc)

        pcm = self._decode_to_pcm_s16le(sound_bytes, sr=sr, channels=channels)
        samples = np.frombuffer(pcm, dtype=np.int16)
        samples_per_chunk = sr * 30
        total = len(samples)
        num_full = total // samples_per_chunk
        rem = total % samples_per_chunk

        chunks: List[SoundChunkData] = []
        for idx in range(num_full):
            s = idx * samples_per_chunk
            e = s + samples_per_chunk
            chunk_pcm = samples[s:e].astype(np.int16).tobytes()
            webm_bytes = self._encode_mp3(chunk_pcm, sr=sr, channels=channels)
            chunks.append(
                SoundChunkData(
                    data=webm_bytes,
                    start_at=start + timedelta(seconds=idx * 30),
                    sampling_rate_hz=float(sr),
                )
            )
        if rem > 0 and total >= rem:
            s = num_full * samples_per_chunk
            e = s + rem
            chunk_pcm = samples[s:e].astype(np.int16).tobytes()
            webm_bytes = self._encode_mp3(chunk_pcm, sr=sr, channels=channels)
            chunks.append(
                SoundChunkData(
                    data=webm_bytes,
                    start_at=start + timedelta(seconds=num_full * 30),
                    sampling_rate_hz=float(sr),
                )
            )
        return chunks


