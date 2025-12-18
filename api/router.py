from api.endpoints import mms

from fastapi import APIRouter

api_router = APIRouter()

api_router.include_router(mms.router)