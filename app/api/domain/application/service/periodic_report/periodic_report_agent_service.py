from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, Optional, Tuple

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

from app.common import config


class PeriodicReportAgentService:
    def __init__(self) -> None:
        self._endpoint = config.AZURE_PROJECT_ENDPOINT
        self._agent_id = config.AZURE_AGENT_ID
        self._logger = logging.getLogger("periodic.agent")

    def _ensure_config(self) -> None:
        if not self._endpoint or not self._agent_id:
            raise RuntimeError("Azure Agent config missing: set AZURE_PROJECT_ENDPOINT and AZURE_AGENT_ID")

    async def analyze(self, payload: dict) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Calls Azure Agent with the provided payload, expecting 2 JSON blocks in assistant output
        (daily_report_json, analysis_json). Returns (None, None) on failure.
        """
        self._ensure_config()
        project = AIProjectClient(credential=DefaultAzureCredential(), endpoint=self._endpoint)
        agent = project.agents.get_agent(self._agent_id)
        thread = project.agents.threads.create()

        json_content = json.dumps(payload, ensure_ascii=False)
        project.agents.messages.create(thread_id=thread.id, role="user", content=json_content)
        run = await asyncio.to_thread(project.agents.runs.create_and_process, thread_id=thread.id, agent_id=agent.id)
        if run.status == "failed":
            self._logger.error("agent_run_failed", extra={"last_error": run.last_error})
            return None, None

        from azure.ai.agents.models import ListSortOrder
        messages = project.agents.messages.list(thread_id=thread.id, order=ListSortOrder.ASCENDING)
        agent_response_text = ""
        for msg in messages:
            if msg.text_messages and msg.role == "assistant":
                agent_response_text += msg.text_messages[-1].text.value

        return self._parse_agent_response(agent_response_text)

    def _parse_agent_response(self, output_text: str) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        import re
        try:
            json_pattern = r'(\{.*?\})'
            json_blocks = re.findall(json_pattern, output_text, re.DOTALL)
            if len(json_blocks) < 2:
                self._logger.error("agent_output_blocks_insufficient", extra={"count": len(json_blocks)})
                return None, None
            daily_block = self._clean_and_load(json_blocks[0])
            analysis_block = self._clean_and_load(json_blocks[1])
            return daily_block, analysis_block
        except Exception:
            self._logger.exception("agent_parse_error")
            return None, None

    def _clean_and_load(self, raw: str) -> Optional[Dict[str, Any]]:
        raw = raw.replace("```json", "").replace("```", "").strip()
        import re, json
        # Quote unquoted keys
        raw = re.sub(r'([\{\,]\s*)(\s*\w+\s*)(\s*:\s*)', r'\1"\2"\3', raw)
        raw = raw.replace(',\n}', '\n}')
        try:
            return json.loads(raw)
        except Exception:
            self._logger.exception("json_decode_error")
            return None


