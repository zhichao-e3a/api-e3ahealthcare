from api.router import api_router
from core.middleware import install_middleware

import os

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from config.configs import REMOTE_MONGO_CONFIG, TEST_MONGO_CONFIG, LOCAL_MONGO_CONFIG, SQL_CONFIG
from database_manager.database.mongo import MongoDBConnector
from database_manager.database.mysql import SQLDBConnector

mode = os.getenv("MODE")

if mode == "TEST": cfg = TEST_MONGO_CONFIG
elif mode == "LOCAL": cfg = LOCAL_MONGO_CONFIG
elif mode == "REMOTE": cfg = REMOTE_MONGO_CONFIG

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.mongo = MongoDBConnector(cfg)
    app.state.sql   = SQLDBConnector(SQL_CONFIG)
    yield

app = FastAPI(lifespan=lifespan)

install_middleware(app)

app.include_router(api_router)

app.mount("/", StaticFiles(directory="static", html=True))

# uvicorn main:app --host 0.0.0.0 --port 8502