from api.endpoints import mms, pipeline

from fastapi import APIRouter

api_router = APIRouter()

api_router.include_router(mms.router)
api_router.include_router(pipeline.router)
