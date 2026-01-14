from utils.common import patch_job
from utils.query import df_to_records
from schemas.records import assert_records_match_schema
from database_manager.database.mongo import MongoDBConnector
from database_manager.database.mysql import SQLDBConnector
from database_manager.database.queries import REBUILD_HISTORICAL, RECRUITED

import anyio
import pandas as pd

from datetime import datetime

OUTLIERS = {"18610926757", "18997836855"}

async def run_rebuild_job(job_id: str, mongo: MongoDBConnector, sql: SQLDBConnector) -> None:

    await patch_job(mongo, job_id, status="running", result=None, error=None)

    try:
        await rebuild(mongo=mongo, sql=sql)
        await patch_job(mongo, job_id, status="succeeded", result=None, error=None)

    except Exception as e:
        await patch_job(mongo, job_id, status="failed", result=None, error=str(e))

async def rebuild(mongo: MongoDBConnector, sql: SQLDBConnector):

    for c in ["RECORDS_RAW", "RECORDS_FILT", "RECORDS_PRED"]:

        n_del = await mongo.delete_all_documents(coll_name=c)

        print(f"[{c}] {n_del} DOCUMENTS DELETED")

    for c in ['RECORDS_RAW']:

        watermark_log = {
            'pipeline_name' : c,
            'last_utime'    : '2000-01-01 00:00:00',
        }

        await mongo.upsert_documents_hashed(
            coll_name   = 'WATERMARKS',
            records     = [watermark_log],
            id_fields   = ["pipeline_name"]
        )

    print(f"[WATERMARKS] UPDATED TO 2000-01-01 00:00:00")

    print()

    print(f"OUTLIERS (RECRUITED PATIENTS BUT MARKED AS HISTORICAL): {OUTLIERS}")

    hist_df = await anyio.to_thread.run_sync(lambda: sql.query_to_dataframe(query=REBUILD_HISTORICAL))

    hist_df = hist_df[~hist_df["mobile"].isin(OUTLIERS)]

    hist_df["origin"] = "HIST"

    print(f"QUERIED FROM NAVICAT ({len(hist_df)} MEASUREMENTS)")

    recruited_patients = await mongo.get_all_documents(
        coll_name   = "METADATA_REC",
        query       = {'processed': False},
        projection  = {'_id': 0, 'mobile': 1}
    )

    recruited_mobiles = [i['mobile'] for i in recruited_patients]

    print(f"QUERIED FROM `METADATA_REC` ({len(recruited_mobiles)} PATIENTS)")

    query_string = ",".join(recruited_mobiles)
    custom_query = RECRUITED.format(
        start="'2025-03-01 00:00:00'",
        end=f"'{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}'",
        numbers=query_string
    )

    rec_df = await anyio.to_thread.run_sync(lambda: sql.query_to_dataframe(query=custom_query))

    rec_df["origin"] = "REC"

    print(f"QUERIED FROM NAVICAT ({len(rec_df)} MEASUREMENTS)")

    df = pd.concat([rec_df, hist_df], ignore_index=True)

    records = await df_to_records(df)

    assert_records_match_schema(records, record_type="RAW")

    for m in recruited_mobiles:

        await mongo.patch_document(
            coll_name   = "METADATA_REC",
            query       = {'mobile': m},
            set_fields  = {'processed': True}
        )

        print(f"PATIENT {m} MARKED AS PROCESSED")

    if len(df) > 0:

        await mongo.upsert_documents_hashed(
            coll_name   = 'RECORDS_RAW',
            records     = records
        )

        print(f"UPSERTED TO 'RECORDS_RAW' ({len(records)} RECORDS)")

    else:
        print(f"NO RECORDS")