#!/usr/bin/env python3
import os
import asyncio
from fastapi import FastAPI

from app.common.logging import configure_logging
from app.api.domain.presentation.router.session_router import Router as SessionRouter
from app.api.domain.presentation.router.brainwave_router import Router as BrainwaveRouter
from app.api.common.exception.exception_handler import register_exception_handlers
from app.api.common.middleware.auth_middleware import AuthMiddleware
from app.api.common.dependencies import container as di_container
from app.api.common.kafka_consumers import KafkaConsumerOrchestrator
from app.api.common.security.token_provider import TokenProvider
from app.common import config

from app.api.domain.worker.brainwave_chunk_splitter.__main__ import run as run_splitter  # type: ignore
from app.api.domain.worker.brainwave_analyzer.__main__ import run as run_analyzer  # type: ignore


configure_logging("fastapi-api")

app = FastAPI()

app.container = di_container
app.state.container = di_container

app.add_middleware(AuthMiddleware, token_provider=TokenProvider())
register_exception_handlers(app)

app.include_router(SessionRouter)
app.include_router(BrainwaveRouter)

_consumers = KafkaConsumerOrchestrator(container=di_container)

@app.on_event("startup")
async def _start_consumers() -> None:
    producer = getattr(di_container, "kafka_producer", None)
    if producer is not None and hasattr(producer, "start"):
        try:
            await producer.start()
        except Exception:
            pass
    if config.KAFKA_ENABLED:
        _consumers.start_all()
        app.state.worker_tasks = [
            asyncio.create_task(run_splitter()),
            asyncio.create_task(run_analyzer()),
        ]
        await _consumers.wait_ready()

        print("AI-sleep-service ready to serve!")

@app.on_event("shutdown")
async def _stop_consumers() -> None:
    if config.KAFKA_ENABLED:
        _consumers.stop_all()
        tasks = getattr(app.state, "worker_tasks", [])
        for t in tasks:
            t.cancel()
            try:
                await t
            except Exception:
                pass
    producer = getattr(di_container, "kafka_producer", None)
    if producer is not None and hasattr(producer, "stop"):
        try:
            await producer.stop()
        except Exception:
            pass

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run("api.main:app", host="0.0.0.0", port=port)
