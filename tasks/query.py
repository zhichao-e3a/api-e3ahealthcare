from utils.common import patch_job
from utils.query import async_process_df, extract_gest_age
from schemas.records import assert_records_match_schema
from database_manager.database.mongo import MongoDBConnector
from database_manager.database.mysql import SQLDBConnector
from database_manager.database.queries import RECRUITED

import anyio
import pandas as pd

from datetime import datetime

async def run_query_job(job_id: str, mongo: MongoDBConnector, sql: SQLDBConnector) -> None:

    await patch_job(mongo, job_id, status="running", result=None, error=None)

    try:
        await query(mongo=mongo, sql=sql)
        await patch_job(mongo, job_id, status="succeeded", result=None, error=None)

    except Exception as e:
        await patch_job(mongo, job_id, status="failed", result=None, error=str(e))


async def query(mongo: MongoDBConnector, sql: SQLDBConnector) -> None:

    curr_watermark = await mongo.get_all_documents(
        coll_name   = "WATERMARKS",
        query       = {"_id": "SQL"},
        projection  = {
            "_id": 0,
            "last_utime": 1
        }
    )

    last_utime = curr_watermark[0]['last_utime']

    print(f"[H] WATERMARK RETRIEVED ({last_utime})")

    recruited_patients = await mongo.get_all_documents(
        coll_name="patients_unified",
        query={
            'type'  : 'rec',
            'add'   : None
        },
        projection={
            '_id': 0,
            'mobile': 1
        }
    )

    recruited_mobiles = [i['mobile'] for i in recruited_patients]

    print(f"[R] QUERIED FROM `patients_unified` ({len(recruited_mobiles)} PATIENTS)")

    query_string = ",".join(recruited_mobiles)
    custom_query = RECRUITED.format(
        start="'2025-03-01 00:00:00'",
        end=f"'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'",
        numbers=query_string
    )

    custom_query += f" WHERE r.utime > '{last_utime}'"

    rec_df = await anyio.to_thread.run_sync(lambda: sql.query_to_dataframe(query=custom_query))

    if len(rec_df) == 0:
        print(f"[A] NO RECORDS")
        return

    rec_df["origin"] = "REC"

    print(f"[R] QUERIED FROM NAVICAT ({len(rec_df)} MEASUREMENTS)")

    rec_df["measurement_date"] = (
        pd.to_datetime(rec_df["start_ts"], unit="s", utc=True)
        .dt.tz_convert("Asia/Singapore")
        .dt.strftime("%Y-%m-%d %H:%M:%S")
    )

    rec_df["start_test_ts"] = (
        pd.to_datetime(rec_df["start_test_ts"], unit="s", utc=True, errors="coerce")
        .dt.tz_convert("Asia/Singapore")
        .dt.strftime("%Y-%m-%d %H:%M:%S")
    )

    rec_df["sql_utime"] = (
        pd.to_datetime(rec_df["utime"], unit="s", utc=True, errors="coerce")
        .dt.tz_convert("Asia/Singapore")
        .dt.strftime("%Y-%m-%d %H:%M:%S")
    )

    # UC, FHR, FMov measurements not ordered yet
    uc_results, fhr_results, fmov_results = await async_process_df(rec_df)

    print(f"[A] DOWNLOADED UC, FHR, FMOV DATA")

    sorted_uc_list = sorted(uc_results, key=lambda x: x[0])
    sorted_fhr_list = sorted(fhr_results, key=lambda x: x[0])
    sorted_fmov_list = sorted(fmov_results, key=lambda x: x[0])

    rec_df["uc_str"]    = [x[1] for x in sorted_uc_list]
    rec_df["fhr_str"]   = [x[1] for x in sorted_fhr_list]
    rec_df["fmov_str"]  = [x[1] for x in sorted_fmov_list]

    rec_df["uc"]    = rec_df["uc_str"].str.split("\n")
    rec_df["fhr"]   = rec_df["fhr_str"].str.split("\n")
    rec_df["fmov"]  = rec_df["fmov_str"].where(rec_df["fmov_str"].astype(bool), None).str.split("\n")

    rec_df["gest_age"] = rec_df.apply(
        lambda r: extract_gest_age(r["conclusion"], r["basic_info"]),
        axis=1
    ).astype("Int64")

    rec_df.rename(columns={"id": "_id"}, inplace=True)

    rec_df.drop(
        columns=[
            "start_ts",
            "contraction_url",
            "hb_baby_url",
            "raw_fetal_url",
            "basic_info",
            "conclusion",
            "expected_born_date",
            "end_born_ts",
            "utime",
            "uc_str",
            "fhr_str",
            "fmov_str"
        ],
        inplace=True,
        errors="ignore"
    )

    records = rec_df.to_dict(orient="records")

    assert_records_match_schema(records, record_type="RAW")

    await mongo.upsert_documents_hashed(
        coll_name='RAW_RECORDS',
        records=records
    )

    print(f"[A] UPSERTED TO 'RAW_RECORDS' ({len(records)} RECORDS)")

    latest_utime = pd.to_datetime(rec_df["sql_utime"]).max().strftime("%Y-%m-%d %H:%M:%S")

    watermark_log = {
        "pipeline_name" : "SQL",
        "last_utime"    : latest_utime
    }

    # Upsert watermark to MongoDB
    await mongo.upsert_documents_hashed(
        coll_name   = 'WATERMARKS',
        records     = [watermark_log],
        id_fields   = ["pipeline_name"]
    )

    print(f"[H] UPSERTED WATERMARK ({latest_utime})")