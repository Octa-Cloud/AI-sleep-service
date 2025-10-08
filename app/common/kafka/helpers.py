from __future__ import annotations


def build_key(session_no: int, trace_id: str) -> str:
    return f"{int(session_no)}:{str(trace_id)}"


def build_headers(trace_id: str, session_no: int, msg: str) -> dict[str, str]:
    return {"trace_id": str(trace_id), "session_no": str(int(session_no)), "version": "1", "content-type": msg}


