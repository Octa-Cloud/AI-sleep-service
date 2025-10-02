from __future__ import annotations

import os
import threading
import time

_LOCK = threading.Lock()
_LAST_MS = 0
_SEQUENCE = 0

# Snowflake-like TSID: 41-bit time, 10-bit node, 12-bit sequence
_EPOCH_MS = int(time.mktime((2020, 1, 1, 0, 0, 0, 0, 0, 0))) * 1000
_NODE_ID = int(os.getenv("NODE_ID", "1")) & 0x3FF


def generate_tsid() -> int:
    global _LAST_MS, _SEQUENCE
    with _LOCK:
        now_ms = int(time.time() * 1000)
        if now_ms < _LAST_MS:
            now_ms = _LAST_MS
        if now_ms == _LAST_MS:
            _SEQUENCE = (_SEQUENCE + 1) & 0xFFF
            if _SEQUENCE == 0:
                while int(time.time() * 1000) <= _LAST_MS:
                    time.sleep(0.000001)
                now_ms = int(time.time() * 1000)
        else:
            _SEQUENCE = 0
        _LAST_MS = now_ms

        ts_part = (now_ms - _EPOCH_MS) & ((1 << 41) - 1)
        node_part = _NODE_ID & 0x3FF
        seq_part = _SEQUENCE & 0xFFF
        return (ts_part << 22) | (node_part << 12) | seq_part


