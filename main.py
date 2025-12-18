from api.router import api_router
from core.middleware import install_middleware

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from config.configs import REMOTE_MONGO_CONFIG
from database_manager.database.mongo import MongoDBConnector

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.mongo = MongoDBConnector(REMOTE_MONGO_CONFIG)
    yield

app = FastAPI(lifespan=lifespan)

install_middleware(app)

app.include_router(api_router)

app.mount("/", StaticFiles(directory="static", html=True))

# uvicorn main:app --host 0.0.0.0 --port 8502