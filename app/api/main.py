#!/usr/bin/env python3
import os
from fastapi import FastAPI

from app.common.logging import configure_logging
from app.api.domain.presentation.router.session_router import Router as SessionRouter
from app.api.domain.presentation.router.brainwave_router import Router as BrainwaveRouter
from app.api.common.exception.exception_handler import register_exception_handlers
from app.api.common.middleware.auth_middleware import AuthMiddleware
from app.api.common.dependencies import container as di_container

import app.api.domain.presentation.router.session_router as session_router_module
import app.api.domain.presentation.router.brainwave_router as brainwave_router_module
import app.api.domain.application.pipeline.brainwave_analyze.pipeline as brainwave_pipeline_module
from app.api.common.security.token_provider import TokenProvider


configure_logging("fastapi-api")

app = FastAPI()

# Simple Python DI container
app.container = di_container
app.state.container = di_container

# Middleware & exception handlers
app.add_middleware(AuthMiddleware, token_provider=TokenProvider())
register_exception_handlers(app)

# Routers
app.include_router(SessionRouter)
app.include_router(BrainwaveRouter)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run("api.main:app", host="0.0.0.0", port=port)
