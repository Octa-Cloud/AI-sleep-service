from __future__ import annotations

import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import pytest


from app.api.domain.application.service.sound.sound_analyzer_service import SoundAnalyzerService
from app.api.domain.domain.entity.analyzed_data_entity import SoundEventType
from app.api.domain.domain.vo.chunked_data_value_object import SoundChunkData


def _ffmpeg_to_mp3_bytes(src_path: Path, *, sr: int = 16000, channels: int = 1, bitrate: str = "64k") -> bytes:
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(src_path),
        "-ac",
        str(channels),
        "-ar",
        str(sr),
        "-f",
        "mp3",
        "-c:a",
        "libmp3lame",
        "-b:a",
        bitrate,
        "-",
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return res.stdout


def _detect_labels(mp3_bytes: bytes, *, top_n: int = 5, threshold: float = 0.7) -> List[str]:
    analyzer = SoundAnalyzerService()
    chunk = SoundChunkData(data=mp3_bytes, start_at=datetime.now(timezone.utc), sampling_rate_hz=16000.0)
    detections = analyzer.analyze_chunk_to_detections(chunk, top_n=top_n, confidence_threshold=threshold)
    return [d["sound"] for d in detections]


def _labels_to_event_types(labels: List[str]) -> List[SoundEventType | None]:
    analyzer = SoundAnalyzerService()
    return [analyzer._label_to_event_type(lbl) for lbl in labels]  # type: ignore[attr-defined]


def _has_snore_label(labels: List[str]) -> bool:
    types = _labels_to_event_types(labels)
    return any(t == SoundEventType.SNORE for t in types)


def _has_snore_or_mouth_breathing(labels: List[str]) -> bool:
    types = _labels_to_event_types(labels)
    return any(t in (SoundEventType.SNORE, SoundEventType.MOUTH_BREATHING) for t in types)


def _tests_root() -> Path:
    for p in Path(__file__).resolve().parents:
        if p.name == "tests" and p.parent.name == "app":
            return p
    raise AssertionError("Cannot locate app/tests root")


@pytest.mark.parametrize("segment", [f"snoring_segment_{i:03d}.webm" for i in range(1, 22)])
def test_snoring_segments_have_any_snore_detection(segment: str):
    base = _tests_root() / "environment" / "data" / "sound"
    src = base / segment
    assert src.exists(), f"missing test file: {segment}"

    mp3 = _ffmpeg_to_mp3_bytes(src)
    labels = _detect_labels(mp3)
    assert _has_snore_or_mouth_breathing(labels), f"{segment}: neither SNORE nor MOUTH_BREATHING detected"


def _sound_dir() -> Path:
    return _tests_root() / "environment" / "data" / "sound"


# Note: Local individual sound-effect tests removed per requirement; mixed-only tests cover detection quality


