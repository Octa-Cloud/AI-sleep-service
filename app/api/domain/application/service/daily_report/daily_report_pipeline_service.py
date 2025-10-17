from __future__ import annotations

from typing import List

from app.api.domain.application.service.daily_report.daily_report_agent_service import DailyReportAgentService

try:
    from app.common.kafka.dto import report_pb2 as rp  # type: ignore
except Exception:  # pragma: no cover
    rp = None  # type: ignore


class DailyReportPipelineService:
    def __init__(self, agent_service: DailyReportAgentService) -> None:
        self._agent = agent_service

    async def build_persist_request(self, dri: "rp.DailyReportInput") -> "rp.DailyReportPersistRequest":  # type: ignore[name-defined]
        session_no = int(dri.session_no)
        user_no = int(dri.user_no)
        created_at_ms = int(dri.created_at_ms)
        sleep_date_str = str(dri.sleep_date)

        payload = {
            "instruction": "Agent의 내장된 '버전 1: ALL-IN-ONE' 지침을 따라 모든 계산 및 분석을 수행하고 4개의 JSON 블록을 출력하십시오. 총 수면 시간 및 비율 계산 시 Wakeup(0)은 제외하고 1, 2, 3, 4, 5만 사용하십시오.",
            "sleep_session_no": session_no,
            "user_no": user_no,
            "predicted_classes_array": [int(it.level) for it in dri.levels],
        }
        b1, b2, b3, b4 = await self._agent.analyze(payload)

        deep_min = int((b1 or {}).get("deep_sleep_minutes", 0)) if b1 else 0
        light_min = int((b1 or {}).get("light_sleep_minutes", 0)) if b1 else 0
        rem_min = int((b1 or {}).get("rem_sleep_minutes", 0)) if b1 else 0
        deep_ratio = float((b1 or {}).get("deep_sleep_ratio", 0.0)) if b1 else 0.0
        light_ratio = float((b1 or {}).get("light_sleep_ratio", 0.0)) if b1 else 0.0
        rem_ratio = float((b1 or {}).get("rem_sleep_ratio", 0.0)) if b1 else 0.0
        total_min = deep_min + light_min + rem_min
        memo = (b2 or {}).get("memo", "") if b2 else ""
        score = int((b2 or {}).get("score", 0)) if b2 else 0

        details_pb: List["rp.AnalysisDetail"] = []  # type: ignore[name-defined]
        if b3 and b4:
            steps_for_detail: List["rp.AnalysisStep"] = []  # type: ignore[name-defined]
            for s in (b4 or []):
                if isinstance(s, dict):
                    steps_for_detail.append(rp.AnalysisStep(step_index=int(s.get("step_index", 0)), content=str(s.get("content", ""))))  # type: ignore[attr-defined]
            try:
                diff = getattr(rp.Difficulty, str(b3.get("difficulty", "EASY")))  # type: ignore[attr-defined]
            except Exception:
                diff = rp.Difficulty.EASY  # type: ignore[attr-defined]
            try:
                eff_name = str(b3.get("effect", "MEDIUM"))
                eff = getattr(rp.Effect, "MEDIUM_E" if eff_name == "MEDIUM" else eff_name)  # type: ignore[attr-defined]
            except Exception:
                eff = rp.Effect.MEDIUM_E  # type: ignore[attr-defined]
            details_pb.append(
                rp.AnalysisDetail(  # type: ignore[attr-defined]
                    title=str(b3.get("title", "")),
                    description=str(b3.get("description", "")),
                    difficulty=diff,
                    effect=eff,
                    steps=steps_for_detail,
                )
            )

        return rp.DailyReportPersistRequest(  # type: ignore[attr-defined]
            session_no=session_no,
            user_no=user_no,
            sleep_date=sleep_date_str,
            created_at_ms=created_at_ms,
            memo=str(memo or ""),
            score=int(score or 0),
            total_sleep_minutes=int(total_min),
            deep_sleep_minutes=int(deep_min),
            light_sleep_minutes=int(light_min),
            rem_sleep_minutes=int(rem_min),
            deep_sleep_ratio=float(deep_ratio),
            light_sleep_ratio=float(light_ratio),
            rem_sleep_ratio=float(rem_ratio),
            details=details_pb,
        )


