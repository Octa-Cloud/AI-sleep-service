import mysql.connector
import pandas as pd
import json
import re
from decimal import Decimal
from datetime import date
from fastapi import FastAPI, HTTPException
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.agents.models import ListSortOrder
import asyncio
from typing import Dict, Any, Tuple, Optional
import os 

app = FastAPI(
    title="Sleep Analyst API",
    description="최신 수면 세션 데이터를 분석하고 JSON만 반환하는 Agent 서비스 API"
)

# Agent 설정 (환경 변수가 아니므로 그대로 유지)
AZURE_PROJECT_ENDPOINT = "https://happy-mgpyzagf-eastus2.services.ai.azure.com/api/projects/happy-mgpyzagf-eastus2_project"
AGENT_ID = "asst_iSkomqUuZXEqzR3BU7Oc3LMG"

# 🚨 수정된 부분: Docker Compose 환경 변수 로드
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "127.0.0.1"), # 기본값은 로컬 유지
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", "1234"),
    "database": os.getenv("DB_NAME", "AI_sleep_service"),
    # Docker 네트워크 통신 시 포트는 3306입니다. (compose 파일에서 3307:3306 매핑됨)
    "port": int(os.getenv("DB_PORT", 3306)) 
}


def get_latest_session_data() -> Tuple[Optional[int], Optional[str], Optional[list], Optional[int], Optional[int], Optional[str]]:
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(sessionid) FROM sleep_chunk")
        latest_session_id = cursor.fetchone()[0]

        if latest_session_id is None:
            conn.close()
            raise ValueError("DB에 유효한 session_id가 없습니다.")

        cursor.execute(f"SELECT user_id FROM sleep_session WHERE session_id = {latest_session_id}")
        user_id_result = cursor.fetchone()
        user_id = user_id_result[0] if user_id_result else None

        # 나이 계산
        cursor.execute(f"SELECT birth FROM User WHERE user_no = {user_id}")
        birth_result = cursor.fetchone()
        user_age = None
        if birth_result and birth_result[0]:
            birth_data = birth_result[0]
            birthday_date = birth_data.date() if hasattr(birth_data, 'date') else birth_data
            today = date.today()
            user_age = today.year - birthday_date.year - ((today.month, today.day) < (birthday_date.month, birthday_date.day))

        df = pd.read_sql(f"""
             SELECT chunk_id, seq_no, started_at, duration_minutes, stage, microwave_grade
             FROM sleep_chunk 
             WHERE session_id = {latest_session_id}
             ORDER BY seq_no ASC
             """, conn)
        conn.close()

        if df.empty:
            raise ValueError(f"세션 ID {latest_session_id}에 대한 데이터가 없습니다.")

        # 데이터 클리닝 로직은 그대로 유지
        for col in df.columns:
            if col not in ['stage', 'started_at']:
                # Decimal 타입을 float으로 변환
                df[col] = df[col].apply(lambda x: float(x) if isinstance(x, Decimal) else x)
        if 'started_at' in df.columns:
            df['started_at'] = df['started_at'].astype(str)

        sleep_date = str(df['started_at'].iloc[0])[:10]
        data_json = df.to_dict(orient='records')

        return latest_session_id, sleep_date, data_json, user_id, user_age, None

    except mysql.connector.Error as err:
        return None, None, None, None, None, f"MySQL 연결 오류: {err}"
    except Exception as e:
        return None, None, None, None, None, str(e)


def parse_agent_response(response_text: str) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[list]]:
    json_pattern = r'(\{[^\{\}]*\}|\[.*?\])'
    json_blocks = re.findall(json_pattern, response_text, re.DOTALL)

    if len(json_blocks) < 3:
        print(f" 파싱 실패: 블록 수 {len(json_blocks)}")
        return None, None, None

    try:
        block1 = json.loads(json_blocks[0].strip())
        block2 = json.loads(json_blocks[1].strip())
        block3 = json.loads(json_blocks[2].strip())
        
        # Agent가 리스트로 보낼 경우를 대비하여 딕셔너리로 변환 (안전 보장)
        if isinstance(block1, list) and len(block1) > 0: block1 = block1[0]
        if isinstance(block2, list) and len(block2) > 0: block2 = block2[0]
        
        return block1, block2, block3
    except json.JSONDecodeError as e:
        print(f" JSON 디코딩 오류: {e}")
        return None, None, None

@app.post("/analyze/daily_report/json")
async def analyze_daily_report_json():
    # DB 저장 로직이 없으므로, Agent의 출력 형태 유지에 대한 지침만 강화합니다.
    
    # DB 필드 목록은 Agent가 생성할 필요 없으므로 주석 처리
    # VALID_REPORT_FIELDS = [...] 

    try:
        # DB 접근은 스레드 풀에서 비동기로 처리
        session_id, sleep_date, data_json_or_error, user_id, user_age, error_msg = await asyncio.to_thread(get_latest_session_data)
        
        if error_msg:
            if "세션 ID" in error_msg or "유효한 session_id" in error_msg:
                raise HTTPException(status_code=404, detail=error_msg)
            else:
                raise HTTPException(status_code=500, detail=error_msg)

        user_prompt = f"""
        당신은 Professional Sleep Analyst입니다.
        아래 세션 ID {session_id}에 대한 JSON 데이터를 분석하여 3개의 JSON 블록을 생성하세요.

        # 필수 데이터
        user_id: {user_id}, sleep_date: {sleep_date}, user_age: {user_age}
        Predicted classes: {json.dumps(data_json_or_error, indent=2)}

        # 지침
        1. 출력은 3개의 JSON 블록만 포함해야 하며, 설명, 주석, Markdown 형식, 도구 사용 명령어는 절대 금지합니다.
        2. **첫 번째 블록(daily_report):** 반드시 단일 JSON 객체(리스트 아님)여야 하며, 'user_age'나 중간 점수 필드('base_sleep_score', 'stage_ratio_score', 'disturbance_penalty')는 포함하지 마십시오.
        3. 세 번째 블록(step)은 반드시 JSON 배열 형태여야 합니다.
        """

        # Agent 호출
        project = AIProjectClient(credential=DefaultAzureCredential(), endpoint=AZURE_PROJECT_ENDPOINT)
        agent = project.agents.get_agent(AGENT_ID)
        thread = project.agents.threads.create()
        project.agents.messages.create(thread_id=thread.id, role="user", content=user_prompt)

        run = await asyncio.to_thread(project.agents.runs.create_and_process, thread_id=thread.id, agent_id=agent.id)
        if run.status == "failed":
            raise HTTPException(status_code=500, detail=f"Agent 실행 실패: {run.last_error}")

        messages = project.agents.messages.list(thread_id=thread.id, order=ListSortOrder.ASCENDING)
        agent_response_text = ""
        for msg in messages:
            if msg.text_messages and msg.role == 'assistant':
                agent_response_text += msg.text_messages[-1].text.value + "\n"

        # JSON 파싱
        daily_report_json, daily_report_analysis_json, daily_report_analysis_step_json = parse_agent_response(agent_response_text)
        if not (daily_report_json and daily_report_analysis_json and daily_report_analysis_step_json):
            raise HTTPException(status_code=500, detail="Agent 응답 JSON 파싱 실패")

        return {
            "status": "success",
            "daily_report_json": daily_report_json,
            "daily_report_analysis_json": daily_report_analysis_json,
            "daily_report_analysis_step_json": daily_report_analysis_step_json
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"시스템 오류: {e}")
