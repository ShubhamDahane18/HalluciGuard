"""FastAPI main application for HalluciGuard."""
from __future__ import annotations
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.utils.logging_config import setup_logging
from backend.api.routes import router
from backend.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown handler."""
    setup_logging()
    import logging
    logger = logging.getLogger(__name__)
    logger.info("HalluciGuard backend starting up…")

    # Pre-load models and build index
    try:
        from backend.services.retriever import build_index
        count = build_index()
        logger.info(f"FAISS index ready with {count} document chunks")
    except Exception as e:
        logger.warning(f"Index build during startup failed: {e}")

    yield
    logger.info("HalluciGuard backend shutting down")


app = FastAPI(
    title="HalluciGuard",
    description="Claim-Level LLM Hallucination Detection & Evaluation Framework",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="")
