from core.database import get_mongo, get_sql
from schemas.job import JobOut
from tasks.merge import run_merge_job
from tasks.rebuild import run_rebuild_job
from tasks.query import run_query_job
from tasks.filter import run_filter_job

import asyncio

from uuid import uuid4
from datetime import datetime
from zoneinfo import ZoneInfo
from fastapi import APIRouter, Response, Depends, HTTPException

from database_manager.database.mongo import MongoDBConnector
from database_manager.database.mysql import SQLDBConnector

router = APIRouter(prefix="/pipeline")


def _to_jobout(job_doc: dict) -> JobOut:

    return JobOut(
        job_id=job_doc["_id"],
        type=job_doc["type"],
        status=job_doc["status"],
        created_at=job_doc["created_at"],
        updated_at=job_doc["updated_at"],
        result=job_doc.get("result"),
        error=job_doc.get("error"),
    )


async def create_job(mongo: MongoDBConnector, job_type: str):

    job_id = uuid4().hex

    doc = {
        "_id": job_id,
        "type": job_type,
        "status": "queued",
        "created_at": datetime.now(ZoneInfo("Asia/Singapore")).strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "updated_at": datetime.now(ZoneInfo("Asia/Singapore")).strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "result": None,
        "error": None,
    }

    await mongo.upsert_documents_hashed(coll_name="JOBS", records=[doc])

    return doc


async def retrieve_job(mongo: MongoDBConnector, job_id: str):

    return await mongo.get_one_document(
        coll_name="JOBS",
        query={"_id": job_id},
        projection={
            "_id": 1,
            "type": 1,
            "status": 1,
            "result": 1,
            "error": 1,
            "created_at": 1,
            "updated_at": 1,
        },
    )


@router.get("/jobs/{job_id}", response_model=JobOut)
async def get_job(job_id: str, response: Response, mongo=Depends(get_mongo)):

    job = await retrieve_job(mongo, job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job["status"] in ("queued", "running"):
        response.status_code = 202
        response.headers["Retry-After"] = "5"
    else:
        response.status_code = 200

    return _to_jobout(job)


@router.post(path="/rebuild", response_model=JobOut)
async def rebuild(
    response: Response,
    mongo: MongoDBConnector = Depends(get_mongo),
    sql: SQLDBConnector = Depends(get_sql),
):

    job = await create_job(mongo, job_type="rebuild")

    asyncio.create_task(run_rebuild_job(job["_id"], mongo=mongo, sql=sql))

    response.status_code = 202
    response.headers["Location"] = f"/pipeline/jobs/{job['_id']}"
    response.headers["Retry-After"] = "5"

    return _to_jobout(job)


@router.post(path="/query", response_model=JobOut)
async def query(
    response: Response,
    mongo: MongoDBConnector = Depends(get_mongo),
    sql: SQLDBConnector = Depends(get_sql),
):

    job = await create_job(mongo, job_type="query")

    asyncio.create_task(run_query_job(job["_id"], mongo=mongo, sql=sql))

    response.status_code = 202
    response.headers["Location"] = f"/pipeline/jobs/{job['_id']}"
    response.headers["Retry-After"] = "5"

    return _to_jobout(job)


@router.post(path="/filter", response_model=JobOut)
async def filter(response: Response, mongo: MongoDBConnector = Depends(get_mongo)):

    job = await create_job(mongo, job_type="filter")

    asyncio.create_task(run_filter_job(job["_id"], mongo=mongo))

    response.status_code = 202
    response.headers["Location"] = f"/pipeline/jobs/{job['_id']}"
    response.headers["Retry-After"] = "5"

    return _to_jobout(job)


@router.post(path="/merge", response_model=JobOut)
async def merge(response: Response, mongo: MongoDBConnector = Depends(get_mongo)):

    job = await create_job(mongo, job_type="merge")

    asyncio.create_task(run_merge_job(job["_id"], mongo=mongo))

    response.status_code = 202
    response.headers["Location"] = f"/pipeline/jobs/{job['_id']}"
    response.headers["Retry-After"] = "5"

    return _to_jobout(job)
