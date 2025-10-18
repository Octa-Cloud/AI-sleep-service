from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Tuple
from time import perf_counter

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.agents.models import ListSortOrder

from app.common import config


class DailyReportAgentService:
    def __init__(self) -> None:
        self._endpoint = config.AZURE_DAILY_PROJECT_ENDPOINT
        self._agent_id = config.AZURE_DAILY_AGENT_ID
        self._logger = logging.getLogger("daily.agent")

    def _ensure_config(self) -> None:
        if not self._endpoint or not self._agent_id:
            raise RuntimeError("Azure Agent config missing: set AZURE_PROJECT_ENDPOINT and AZURE_AGENT_ID")

    async def analyze(self, payload: dict) -> Tuple[Optional[dict], Optional[dict], Optional[dict], Optional[List[dict]]]:
        """
        Calls Azure Agent with the provided payload, expecting 4 JSON blocks in assistant output.
        Returns block1, block2, block3, block4 (last is list) or (None, ...) on failure.
        """
        self._ensure_config()
        started = perf_counter()
        # start

        try:
            project = AIProjectClient(credential=DefaultAzureCredential(), endpoint=self._endpoint)
            agent = project.agents.get_agent(self._agent_id)
            thread = project.agents.threads.create()

            # agent_thread_created

            json_content = json.dumps(payload, ensure_ascii=False)
            project.agents.messages.create(thread_id=thread.id, role="user", content=json_content)
            run = await asyncio.to_thread(project.agents.runs.create_and_process, thread_id=thread.id, agent_id=agent.id)
            if run.status == "failed":
                self._logger.error("agent_run_failed", extra={"last_error": run.last_error})
                return None, None, None, None

            messages = project.agents.messages.list(thread_id=thread.id, order=ListSortOrder.ASCENDING)
            agent_response_text = ""
            msg_count = 0
            for msg in messages:
                if msg.text_messages and msg.role == "assistant":
                    agent_response_text += msg.text_messages[-1].text.value + "\n"
                msg_count += 1
                print('daily=', agent_response_text)

            b1, b2, b3 = self._parse_agent_response(agent_response_text)
            if not (b1 and b2 and isinstance(b3, list)):
                self._logger.error(
                    "agent_parse_invalid_blocks",
                    extra={
                        "has_b1": bool(b1),
                        "has_b2": bool(b2),
                        "b3_is_list": isinstance(b3, list),
                    },
                )
            # done
            return b1, b2, b3
        except Exception:
            self._logger.exception("agent_call_error")
            raise

    def _parse_agent_response(self, response_text: str) -> Tuple[Optional[dict], Optional[dict], Optional[List[dict]]]:
        import re
        try:
            json_pattern = r'(\{[^\{\}]*\}|\[.*?\])'
            json_blocks = re.findall(json_pattern, response_text, re.DOTALL)
            if len(json_blocks) < 3:
                self._logger.error("agent_output_blocks_insufficient", extra={"count": len(json_blocks)})
                return None, None, None
            b1 = json.loads(json_blocks[0].strip())
            b2 = json.loads(json_blocks[1].strip())
            b3 = json.loads(json_blocks[2].strip())
            if isinstance(b1, list) and b1: b1 = b1[0]
            if isinstance(b2, list) and b2: b2 = b2[0]
            if not isinstance(b3, list):
                self._logger.error("agent_block3_not_list")
                return None, None, None
            return b1, b2, b3
        except Exception:
            self._logger.exception("agent_parse_error")
            raise
