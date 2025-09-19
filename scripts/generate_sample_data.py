#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date, timedelta
from typing import Iterable, List, Tuple


EPOCH = datetime(1970, 1, 1)


@dataclass(frozen=True)
class User:
    user_no: int
    name: str
    nickname: str
    email: str
    password: str
    gender: str  # 'MALE' | 'FEMALE'


def daterange(start_date: date, end_date: date) -> Iterable[date]:
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def as_dt(d: date, hh: int, mm: int, ss: int = 0, micros: int = 0) -> datetime:
    return datetime(d.year, d.month, d.day, hh, mm, ss, micros)


def to_sql_datetime(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f").rstrip("0").rstrip(".")


def epoch_microseconds(dt: datetime) -> int:
    return int((dt - EPOCH).total_seconds() * 1_000_000)


def generate_sleep_session_id(user_no: int, d: date) -> int:
    # Deterministic bigint: user_no prefix + YYYYMMDD
    return int(f"{user_no}20{d.year % 100:02d}{d.month:02d}{d.day:02d}")


def generate_analysis_detail_id(session_id: int, index_within_session: int) -> int:
    return 2_000_000_000 + session_id * 10 + index_within_session


def generate_analysis_step_id(detail_id: int, step_index: int) -> int:
    return 3_000_000_000 + detail_id * 10 + step_index


def generate_periodic_report_id(start_dt: datetime, user_no: int, salt: int) -> int:
    return epoch_microseconds(start_dt) + user_no * 1_000 + salt


def sql_escape(value: str) -> str:
    return value.replace("'", "''")


def build_inserts() -> List[str]:
    lines: List[str] = []
    lines.append("-- INSERT-only dataset for mong-ai September 2025\nUSE test;")

    # 1) User
    user = User(
        user_no=1,
        name="홍길동",
        nickname="길동",
        email="gildong@example.com",
        password="password-hash",
        gender="MALE",
    )
    lines.append(
        "\n-- Users\n"
        f"INSERT INTO users (user_no, name, nickname, email, password, gender) VALUES\n"
        f"  ({user.user_no}, '{sql_escape(user.name)}', '{sql_escape(user.nickname)}', '{sql_escape(user.email)}', '{sql_escape(user.password)}', '{user.gender}');"
    )

    # 2) Daily sleep sessions and reports for 2025-09-01..2025-09-30, 21:00-05:00 (8h)
    start = date(2025, 9, 1)
    end = date(2025, 9, 30)

    # Sleep sessions
    lines.append("\n-- Sleep sessions (21:00-05:00, 8 hours)\nINSERT INTO sleep_sessions (sleep_session_no, user_no, finished_at, created_at) VALUES")
    ss_values: List[str] = []
    # Daily reports
    dr_values: List[str] = []
    # Sleep time details (use ratios 20/60/20 => 96/288/96 minutes)
    std_deep, std_light, std_rem = 96, 288, 96
    std_deep_r, std_light_r, std_rem_r = 20.0, 60.0, 20.0
    std_details_values: List[str] = []

    # Analysis (2 details per session, 3 steps each, all Korean)
    ad_values: List[str] = []
    astep_values: List[str] = []

    # Analyzed events/levels (events: 30 per session; levels: 30-second intervals across session)
    ase_values: List[str] = []  # analyzed_sound_events
    asl_values: List[str] = []  # analyzed_sleep_levels

    for d in daterange(start, end):
        session_id = generate_sleep_session_id(user.user_no, d)

        start_dt = as_dt(d, 21, 0, 0, 0)
        finish_dt = as_dt(d + timedelta(days=1), 5, 0, 0, 0)

        ss_values.append(
            f"  ({session_id}, {user.user_no}, '{to_sql_datetime(finish_dt)}', '{to_sql_datetime(start_dt)}')"
        )

        dr_values.append(
            f"  ({session_id}, '자동 생성 리포트 {d:%Y-%m-%d}', {user.user_no})"
        )

        std_details_values.append(
            "  ("
            f"{session_id}, {std_deep}, {std_light}, {std_rem}, {std_deep_r:.1f}, {std_light_r:.1f}, {std_rem_r:.1f}"
            ")"
        )

        # Analysis Details (2 per session)
        detail_specs: List[Tuple[str, str, str, str]] = [
            ("수면 위생 개선", "늦은 야식과 카페인을 줄여 숙면을 돕습니다", "EASY", "MEDIUM"),
            ("수면 일정 고정", "규칙적인 취침/기상 시간으로 수면의 질을 높입니다", "MEDIUM", "HIGH"),
        ]

        for idx, (title, desc, difficulty, effect) in enumerate(detail_specs, start=1):
            detail_id = generate_analysis_detail_id(session_id, idx)
            ad_values.append(
                "  ("
                f"{detail_id}, {session_id}, '{sql_escape(title)}', '{sql_escape(desc)}', '{difficulty}', '{effect}'"
                ")"
            )

            steps = [
                "자기 전 전자기기 사용을 30분 줄이기",
                "오후 3시 이후 카페인 섭취 피하기",
                "매일 같은 시간에 기상 알람 설정하기",
            ]
            for s_idx, content in enumerate(steps, start=1):
                astep_id = generate_analysis_step_id(detail_id, s_idx)
                astep_values.append(
                    f"  ({astep_id}, {detail_id}, {s_idx}, '{sql_escape(content)}')"
                )

        # Analyzed sound events (30 per session), cycle through ALL enum types
        # Enum per schema: 'SNORE','BABY_CRYING','COUGH','MOUTH_BREATHING','ANIMAL_NOISE','CAR_HORN'
        event_types = [
            "SNORE",
            "BABY_CRYING",
            "COUGH",
            "MOUTH_BREATHING",
            "ANIMAL_NOISE",
            "CAR_HORN",
        ]
        duration_minutes = int((finish_dt - start_dt).total_seconds() // 60)
        interval_minutes = max(1, duration_minutes // 30)
        for i in range(30):
            ts = start_dt + timedelta(minutes=i * interval_minutes, microseconds=i)
            tsid = epoch_microseconds(ts)
            ev = event_types[i % len(event_types)]
            ase_values.append(
                f"  ({tsid}, {session_id}, '{ev}', '{to_sql_datetime(ts)}')"
            )

        # Analyzed sleep levels at 30-second granularity for entire session duration
        # Build a repeating pattern within valid range 0..6
        level_pattern = [0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1]
        total_seconds = int((finish_dt - start_dt).total_seconds())
        points = total_seconds // 30  # one sample every 30 seconds
        for i in range(points):
            lvl = level_pattern[i % len(level_pattern)]
            ts = start_dt + timedelta(seconds=i * 30, microseconds=i)
            tsid = epoch_microseconds(ts)
            asl_values.append(
                f"  ({tsid}, {session_id}, {lvl}, '{to_sql_datetime(ts)}')"
            )

    lines.append(
        ",\n".join(ss_values) + ";"
    )

    lines.append(
        "\n-- Daily reports\nINSERT INTO daily_reports (sleep_session_no, memo, user_no) VALUES\n"
        + ",\n".join(dr_values)
        + ";"
    )

    lines.append(
        "\n-- Sleep time details (minutes and ratios)\n"
        "INSERT INTO sleep_time_details (sleep_session_no, deep_sleep_minutes, light_sleep_minutes, rem_sleep_minutes, deep_sleep_ratio, light_sleep_ratio, rem_sleep_ratio) VALUES\n"
        + ",\n".join(std_details_values)
        + ";"
    )

    lines.append(
        "\n-- Analysis details (Korean)\n"
        "INSERT INTO analysis_details (analysis_detail_no, sleep_session_no, title, description, difficulty, effect) VALUES\n"
        + ",\n".join(ad_values)
        + ";"
    )

    lines.append(
        "\n-- Analysis steps (3 per detail)\n"
        "INSERT INTO analysis_steps (analysis_step_no, analysis_detail_no, step_index, content) VALUES\n"
        + ",\n".join(astep_values)
        + ";"
    )

    lines.append(
        "\n-- Analyzed sound events (30 per session)\n"
        "INSERT INTO analyzed_sound_events (analyzed_sound_event_no, sleep_session_no, event, recorded_at) VALUES\n"
        + ",\n".join(ase_values)
        + ";"
    )

    lines.append(
        "\n-- Analyzed sleep levels (30 per session)\n"
        "INSERT INTO analyzed_sleep_levels (analyzed_sleep_level_no, sleep_session_no, level, recorded_at) VALUES\n"
        + ",\n".join(asl_values)
        + ";"
    )

    # 3) Periodic reports (weekly for each week in Sep, and monthly)
    lines.append("\n-- Periodic reports (weekly x4 and monthly x1)\n")
    # Weeks: 9/1, 9/8, 9/15, 9/22
    week_starts = [date(2025, 9, 1), date(2025, 9, 8), date(2025, 9, 15), date(2025, 9, 22)]
    pr_values: List[str] = []
    # Monthly start
    monthly_start = date(2025, 9, 1)

    # Weekly reports
    for idx, ws in enumerate(week_starts):
        start_dt_w = as_dt(ws, 0, 0, 0)
        pr_id = generate_periodic_report_id(start_dt_w, user.user_no, salt=100 + idx)
        # Determine end date within September
        week_end = min(ws + timedelta(days=6), date(2025, 9, 30))
        session_count = (week_end - ws).days + 1
        pr_values.append(
            f"  ({pr_id}, {user.user_no}, {session_count}, 'WEEKLY', '{ws:%Y-%m-%d}')"
        )

    # Monthly report
    pr_monthly_id = generate_periodic_report_id(as_dt(monthly_start, 0, 0, 0), user.user_no, salt=1)
    pr_values.append(
        f"  ({pr_monthly_id}, {user.user_no}, 30, 'MONTHLY', '{monthly_start:%Y-%m-%d}')"
    )

    lines.append(
        "INSERT INTO periodic_reports (periodic_report_no, user_no, sleep_session_count, duration_type, period_started_at) VALUES\n"
        + ",\n".join(pr_values)
        + ";"
    )

    # 4) Score prediction points (6 per periodic report, increasing)
    spp_values: List[str] = []
    # Build for weekly
    for idx, ws in enumerate(week_starts):
        start_dt_w = as_dt(ws, 0, 0, 0)
        pr_id = generate_periodic_report_id(start_dt_w, user.user_no, salt=100 + idx)
        base_score = 60 + idx * 2
        for k in range(6):
            d_index = ws + timedelta(days=min(k, (date(2025, 9, 30) - ws).days))
            score = base_score + k * 2
            spp_id = pr_id + k + 1
            spp_values.append(
                f"  ({spp_id}, {pr_id}, '{d_index:%Y-%m-%d}', {score})"
            )

    # Monthly
    base_score = 65
    for k in range(6):
        d_index = date(2025, 9, 1) + timedelta(days=k * 5)
        score = base_score + k * 3
        spp_id = pr_monthly_id + k + 1
        spp_values.append(
            f"  ({spp_id}, {pr_monthly_id}, '{d_index:%Y-%m-%d}', {score})"
        )

    lines.append(
        "\nINSERT INTO score_prediction_points (score_prediction_point_no, periodic_report_no, date_index, score) VALUES\n"
        + ",\n".join(spp_values)
        + ";"
    )

    return lines


def main() -> None:
    sql_lines = build_inserts()
    out_path = "generated_sample_data.sql"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(sql_lines) + "\n")
    print(f"Wrote {out_path} with {sum(1 for _ in open(out_path, 'r', encoding='utf-8'))} lines")


if __name__ == "__main__":
    main()


