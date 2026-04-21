import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers.dogs import router as dogs_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

app = FastAPI(title="Doggies API", version="1.0.0")

# CORS — allow requests from the Next.js dev server (Phase 2)
# extend with prod URL in Phase 3
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.include_router(dogs_router, prefix="/api")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
