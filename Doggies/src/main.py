import logging

from fastapi import FastAPI

logger = logging.getLogger(__name__)

app = FastAPI(title="Doggies", description="AI-powered dog adoption bot for Koh Phangan")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
