from database_manager.database.mongo import MongoDBConnector

from datetime import datetime
from zoneinfo import ZoneInfo

async def patch_job(mongo: MongoDBConnector, job_id: str, **fields) -> None:

    fields["updated_at"] = datetime.now(ZoneInfo("Asia/Singapore")).strftime("%Y-%m-%d %H:%M:%S")

    await mongo.patch_document(
        coll_name   = "JOBS",
        query       = {"_id": job_id},
        set_fields  = fields
    )