import mysql.connector
import pandas as pd
import json
import re
from decimal import Decimal
from datetime import date, datetime
from fastapi import FastAPI, HTTPException
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.agents.models import ListSortOrder
import asyncio
from typing import Dict, Any, Tuple, Optional, List
import os 
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Sleep Analyst API",
    description="최신 수면 세션 데이터를 분석하고 JSON만 반환하는 Agent 서비스 API"
)

# Agent 설정 (환경 변수에서 로드)
# 실제 환경 변수 설정에 따라 AZURE_PROJECT_ENDPOINT와 AGENT_ID를 적절히 설정해야 합니다.
AZURE_PROJECT_ENDPOINT = os.getenv("AZURE_PROJECT_ENDPOINT") 
AGENT_ID = "asst_iSkomqUuZXEqzR3BU7Oc3LMG" 

# Docker Compose 환경 변수 로드
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "127.0.0.1"), 
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", "1234"),
    "database": os.getenv("DB_NAME", "AI_sleep_service"),
    "port": int(os.getenv("DB_PORT", 3306)) 
}

def get_latest_session_data() -> Tuple[Optional[int], Optional[str], Optional[list], Optional[int], Optional[int], Optional[str], Optional[str]]:
    """DB에서 최신 세션의 수면 레벨 데이터와 사용자 정보를 가져옵니다."""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # 1. 최신 세션 ID, user_id, finished_at 가져오기
        cursor.execute("""
            SELECT 
                t1.session_id, 
                t1.user_id, 
                t1.finished_at
            FROM sleep_session t1
            ORDER BY t1.finished_at DESC
            LIMIT 1
        """)
        session_info_result = cursor.fetchone()

        if session_info_result is None:
            conn.close()
            raise ValueError("DB에 유효한 session_id가 없습니다.")

        latest_session_id, user_id, finished_at_dt = session_info_result
        finished_at = str(finished_at_dt) if finished_at_dt else None
        
        # 2. 나이 계산
        cursor.execute(f"SELECT birth FROM User WHERE user_no = {user_id}")
        birth_result = cursor.fetchone()
        user_age = None
        if birth_result and birth_result[0]:
            birth_data = birth_result[0]
            birthday_date = birth_data.date() if hasattr(birth_data, 'date') else birth_data
            today = date.today()
            user_age = today.year - birthday_date.year - ((today.month, today.day) < (birthday_date.month, birthday_date.day))

        # 3. 수면 레벨 (level/stage) 데이터 가져오기 (960개 배열)
        df = pd.read_sql(f"""
             SELECT stage 
             FROM sleep_chunk 
             WHERE session_id = {latest_session_id}
             ORDER BY seq_no ASC
             """, conn)
        conn.close()

        if df.empty:
            raise ValueError(f"세션 ID {latest_session_id}에 대한 수면 레벨 데이터가 없습니다.")

        # stage (level) 값만 리스트로 추출
        predicted_classes = df['stage'].tolist()
        
        memo = "요즘 스트레스가 많아 수면 시작이 어려웠습니다. (35세)" # 예시 Memo를 코드 레벨에서 시뮬레이션
        if user_age:
            memo = f"요즘 스트레스가 많아 수면 시작이 어려웠습니다. ({user_age}세)" 

        return latest_session_id, finished_at, predicted_classes, user_id, user_age, memo, None

    except mysql.connector.Error as err:
        return None, None, None, None, None, None, f"MySQL 연결 오류: {err}"
    except Exception as e:
        return None, None, None, None, None, None, str(e)


# Agent 응답 파싱 함수 (4개 JSON 블록에 맞게 유지)
def parse_agent_response(response_text: str) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[List[Dict[str, Any]]]]:
    json_pattern = r'(\{[^\{\}]*\}|\[.*?\])'
    json_blocks = re.findall(json_pattern, response_text, re.DOTALL)

    if len(json_blocks) < 4:
        logging.error(f"JSON 파싱 실패: 블록 수 {len(json_blocks)}. 원본 텍스트:\n{response_text}")
        return None, None, None, None

    try:
        block1 = json.loads(json_blocks[0].strip()) # sleep_time_details
        block2 = json.loads(json_blocks[1].strip()) # daily_reports
        block3 = json.loads(json_blocks[2].strip()) # analysis_details
        block4 = json.loads(json_blocks[3].strip()) # analysis_steps (배열)
        
        # 안전을 위해 객체 보정
        if isinstance(block1, list) and len(block1) > 0: block1 = block1[0]
        if isinstance(block2, list) and len(block2) > 0: block2 = block2[0]
        if isinstance(block3, list) and len(block3) > 0: block3 = block3[0]
        
        if not isinstance(block4, list): 
             logging.error("BLOCK 4가 리스트(배열) 형태가 아닙니다.")
             return None, None, None, None

        return block1, block2, block3, block4
    except json.JSONDecodeError as e:
        logging.error(f"JSON 디코딩 오류: {e}")
        return None, None, None, None


@app.post("/analyze/daily_report/json")
async def analyze_daily_report_json():
    try:
        # DB 접근은 스레드 풀에서 비동기로 처리
        session_id, finished_at, predicted_classes, user_id, user_age, memo, error_msg = await asyncio.to_thread(get_latest_session_data)
        
        if error_msg:
            if "세션 ID" in error_msg or "유효한 session_id" in error_msg:
                raise HTTPException(status_code=404, detail=error_msg)
            else:
                raise HTTPException(status_code=500, detail=error_msg)

        # 🚨 수정된 부분: Agent에 전달할 데이터를 JSON 객체로 직접 구성하여 메시지 content에 넣습니다.
        # Agent는 이 JSON을 보고 내부 Prompt 로직을 실행합니다.
        input_data_for_agent = {
            "instruction": "Agent의 내장된 '버전 1: ALL-IN-ONE' 지침을 따라 모든 계산 및 분석을 수행하고 4개의 JSON 블록을 출력하십시오. 총 수면 시간 및 비율 계산 시 Wakeup(0)은 제외하고 1, 2, 3, 4, 5만 사용하십시오.",
            "sleep_session_no": session_id,
            "user_no": user_id,
            "finished_at": finished_at,
            "user_age": user_age,
            "Memo": memo,
            "predicted_classes_array": predicted_classes 
        }
        
        # Agent에 메시지 전달 (Prompt 문자열 생성 없이 데이터만 전달)
        project = AIProjectClient(credential=DefaultAzureCredential(), endpoint=AZURE_PROJECT_ENDPOINT)
        agent = project.agents.get_agent(AGENT_ID)
        thread = project.agents.threads.create()
        
        # 데이터를 JSON 문자열로 변환하여 content에 넣습니다.
        json_content = json.dumps(input_data_for_agent, ensure_ascii=False)
        project.agents.messages.create(thread_id=thread.id, role="user", content=json_content)
        logging.info(f"Agent에 데이터 전송 완료: Session ID {session_id}")

        # Run 생성 및 처리
        run = await asyncio.to_thread(project.agents.runs.create_and_process, thread_id=thread.id, agent_id=agent.id)
        if run.status == "failed":
            raise HTTPException(status_code=500, detail=f"Agent 실행 실패: {run.last_error}")

        # 메시지 리스트 가져와서 응답 텍스트 추출
        messages = project.agents.messages.list(thread_id=thread.id, order=ListSortOrder.ASCENDING)
        agent_response_text = ""
        for msg in messages:
            if msg.text_messages and msg.role == 'assistant':
                agent_response_text += msg.text_messages[-1].text.value + "\n"
        
        # JSON 파싱 (4개 블록)
        block1, block2, block3, block4 = parse_agent_response(agent_response_text)
        if not (block1 and block2 and block3 and block4):
            raise HTTPException(status_code=500, detail="Agent 응답 JSON 파싱 실패. 출력 블록의 개수(4개)나 형식을 확인하세요.")

        return {
            "status": "success",
            "sleep_time_details_json": block1,       # BLOCK 1
            "daily_reports_json": block2,            # BLOCK 2
            "analysis_details_json": block3,         # BLOCK 3
            "analysis_steps_array": block4           # BLOCK 4 (배열)
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        logging.error(f"시스템 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=f"시스템 오류: {e}")