from utils.common import patch_job
from utils.filter import async_filter_record
from schemas.records import assert_records_match_schema
from database_manager.database.mongo import MongoDBConnector

import anyio

limiter = anyio.CapacityLimiter(8)

async def run_filter_job(job_id: str, mongo: MongoDBConnector) -> None:

    await patch_job(mongo, job_id, status="running", result=None, error=None)

    try:
        await filter(mongo=mongo)
        await patch_job(mongo, job_id, status="succeeded", result=None, error=None)

    except Exception as e:
        await patch_job(mongo, job_id, status="failed", result=None, error=str(e))


async def filter(mongo: MongoDBConnector) -> None:

    curr_watermark = await mongo.get_all_documents(
        coll_name   = "WATERMARKS",
        query       = {
            "_id": "RECORDS_RAW"
        },
        projection  = {
            "_id"        : 0,
            "last_utime" : 1,
            "last_id"    : 1
        }
    )

    if curr_watermark:

        last_utime  = curr_watermark[0]['last_utime']
        last_id     = curr_watermark[0]['last_id']

        print(f"WATERMARK RETRIEVED ({last_utime}, {last_id})")

        raw_records = mongo.stream_all_documents(
            coll_name = "RECORDS_RAW",
            query={
                "$or": [
                    {"utime": {"$gt": last_utime}},
                    {"utime": last_utime, "_id": {"$gt": last_id}}
                ]
            },
            projection = {
                "ctime"     : 0,
                "doc_hash"  : 0
            },
            sort = [
                ("utime", 1),
                ("_id", 1)
            ]
        )

    else:

        raw_records = mongo.stream_all_documents(
            coll_name = "RECORDS_RAW",
            projection = {
                "ctime"     : 0,
                "doc_hash"  : 0
            },
            sort = [
                ("utime", 1),
                ("_id", 1)
            ]
        )

    all_added   = 0
    all_skipped = 0
    async for batch in raw_records:

        batch_size      = len(batch)
        batch_max_id    = batch[-1]["_id"]
        batch_max_utime = batch[-1]["utime"]

        print(f"[B] {batch_size} BATCH")

        async with anyio.create_task_group() as tg:

            filt_records = []
            async def runner(rec):
                res = await async_filter_record(rec, limiter)
                if res is not None:
                    filt_records.append(res)

            for record in batch:
                tg.start_soon(runner, record)

        batch_skipped = batch_size - len(filt_records)
        print(f"[B] {len(filt_records)} RECORDS BUILT")
        print(f"[B] {batch_skipped} RECORDS SKIPPED")

        all_added   += len(filt_records)
        all_skipped += batch_skipped

        assert_records_match_schema(filt_records, record_type="FILT")

        if len(filt_records) > 0:

            await mongo.upsert_documents_hashed(
                coll_name=f'RECORDS_FILT',
                records=filt_records
            )

            print(f"[B] UPSERTED TO `RECORDS_FILT` ({len(filt_records)} RECORDS)")

        watermark_log = {
            "pipeline_name" : "RECORDS_RAW",
            "last_utime"    : batch_max_utime,
            "last_id"       : batch_max_id,
        }

        # Upsert watermark to MongoDB
        await mongo.upsert_documents_hashed(
            coll_name   = 'WATERMARKS',
            records     = [watermark_log],
            id_fields   = ["pipeline_name"]
        )

        print(f"[B] UPSERTED WATERMARK ({batch_max_utime})")

    print(f"UPSERTED TO `RECORDS_FILT` ({all_added} RECORDS)")
    print(f"{all_skipped} RECORDS SKIPPED")