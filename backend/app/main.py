
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.support import router as support_router
from app.database.base import Base
from app.database.session import engine
from app.database import models


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title="AI Customer Support Agent",
    description="An AI-powered customer support system",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    return {
        "message": "AI Customer Support Agent is running!"
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


app.include_router(
    support_router,
    prefix="/api/v1/support",
    tags=["Customer Support"]
)
