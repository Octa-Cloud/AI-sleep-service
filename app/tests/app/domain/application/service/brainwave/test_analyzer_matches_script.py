from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pytest


def _predict_with_service(edf_path: Path, models_dir: Path) -> list[int]:
    try:
        import mne  # type: ignore
    except Exception:
        pytest.skip("mne not installed")
        raise

    from app.api.domain.application.service.brainwave.brainwave_analyzer_service import BrainwaveAnalyzerService
    from app.api.domain.domain.vo.chunked_data_value_object import BrainwaveChunkData

    os.environ.setdefault("MODELS_DIR", str(models_dir))

    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    try:
        raw.pick_channels(["EEG Fpz-Cz", "EEG Pz-Oz"])  # type: ignore[arg-type]
    except Exception:
        raw.pick_channels(["Fpz-Cz", "Pz-Oz"])  # type: ignore[arg-type]
    raw.filter(l_freq=0.5, h_freq=30.0, fir_design='firwin', verbose=False)
    sfreq = float(raw.info['sfreq'])
    data = raw.get_data().astype(np.float32)

    chunk = BrainwaveChunkData(data=data, start_at=datetime.now(timezone.utc), sampling_rate_hz=sfreq)
    analyzer = BrainwaveAnalyzerService()
    vos = analyzer.analyze([chunk])
    return [int(v.level) for v in vos]


# Expected classes per file (length 20 each)
EXPECTED = {
    "brainwave-10min-ok1.edf":  [0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 5, 5],
    "brainwave-10min-ok2.edf":  [2, 5, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    "brainwave-10min-ok3.edf":  [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2],
    "brainwave-10min-ok4.edf":  [2, 2, 2, 2, 3, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3],
    "brainwave-10min-ok5.edf":  [2, 2, 3, 2, 2, 2, 2, 2, 3, 2, 2, 3, 3, 2, 3, 3, 2, 3, 3, 3],
    "brainwave-10min-ok6.edf":  [2, 2, 2, 2, 2, 2, 3, 2, 3, 4, 3, 2, 2, 3, 3, 2, 2, 2, 2, 2],
    "brainwave-10min-ok7.edf":  [2, 2, 2, 3, 2, 2, 3, 3, 3, 3, 3, 3, 4, 3, 3, 3, 3, 3, 2, 3],
    "brainwave-10min-ok8.edf":  [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    "brainwave-10min-ok9.edf":  [2, 2, 2, 2, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    "brainwave-10min-ok10.edf": [2, 2, 2, 2, 2, 2, 2, 2, 5, 2, 5, 2, 2, 2, 2, 2, 2, 5, 2, 2],
    "brainwave-10min-ok11.edf": [2, 5, 1, 5, 5, 5, 2, 2, 2, 5, 5, 5, 5, 5, 5, 2, 5, 2, 5, 2],
}


@pytest.mark.parametrize("file_name", list(EXPECTED.keys()))
def test_analyzer_ok_files(file_name: str):
    tests_dir = Path(__file__).resolve().parents[5]
    edf_dir = tests_dir / "environment" / "data" / "edf"
    edf_path = edf_dir / file_name
    assert edf_path.exists(), f"EDF not found: {edf_path}"

    models_dir = Path(os.getenv("MODELS_DIR") or (tests_dir.parent / "models"))
    classes = _predict_with_service(edf_path, models_dir)

    # Pretty logs similar to the provided reference output
    files = list(EXPECTED.keys())
    idx = files.index(file_name) + 1
    total = len(files) + 3  # mimic 14 total with other files present in folder
    print(f"\n--- ⏱️ [파일 {idx}/{total}: {file_name}] 분석 시작 ---")
    try:
        import mne  # type: ignore
        raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose=False)
        try:
            raw.pick_channels(["EEG Fpz-Cz", "EEG Pz-Oz"])  # type: ignore[arg-type]
        except Exception:
            raw.pick_channels(["Fpz-Cz", "Pz-Oz"])  # type: ignore[arg-type]
        sfreq = float(raw.info['sfreq'])
        sec = raw.times[-1]
        samples_per_epoch = int(round(30 * sfreq))
        remainder = int(round((sec * sfreq))) % samples_per_epoch
        if remainder != 0:
            print(f"경고: 파일 길이({sec:.2f}초)가 30초의 정수배가 아닙니다. 온전한 30초 에포크만 사용됩니다.")
    except Exception:
        pass
    print(f"  분석된 시간: {len(classes) * 30 / 60:.2f}분 (30초 에포크 {len(classes)}개)")
    print(f"  예측된 수면 단계: {classes[:20]} ...")

    # Verify exact expected sequence
    expected = EXPECTED[file_name]
    assert classes == expected, f"Mismatch for {file_name}: got={classes} expected={expected}"

