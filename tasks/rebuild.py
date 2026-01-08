from utils.common import patch_job
from utils.query import extract_gest_age, async_process_df
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

    for c in ["RAW_RECORDS", "FILT_RECORDS", "MERGED_RECORDS"]:

        n_del = await mongo.delete_all_documents(coll_name=c)

        print(f"[{c}] {n_del} DOCUMENTS DELETED")

    for c in ['SQL', 'RAW_RECORDS']:

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

    df["measurement_date"] = (
        pd.to_datetime(df["start_ts"], unit="s", utc=True)
        .dt.tz_convert("Asia/Singapore")
        .dt.strftime("%Y-%m-%d %H:%M:%S")
    )

    df["start_test_ts"] = (
        pd.to_datetime(df["start_test_ts"], unit="s", utc=True, errors="coerce")
        .dt.tz_convert("Asia/Singapore")
        .dt.strftime("%Y-%m-%d %H:%M:%S")
    )

    df["sql_utime"] = (
        pd.to_datetime(df["utime"], unit="s", utc=True, errors="coerce")
        .dt.tz_convert("Asia/Singapore")
        .dt.strftime("%Y-%m-%d %H:%M:%S")
    )

    # UC, FHR, FMov measurements not ordered yet
    uc_results, fhr_results, fmov_results = await async_process_df(df)

    print(f"DOWNLOADED UC, FHR, FMOV DATA")

    sorted_uc_list      = sorted(uc_results, key=lambda x: x[0])
    sorted_fhr_list     = sorted(fhr_results, key=lambda x: x[0])
    sorted_fmov_list    = sorted(fmov_results, key=lambda x: x[0])

    df["uc_str"]    = [x[1] for x in sorted_uc_list]
    df["fhr_str"]   = [x[1] for x in sorted_fhr_list]
    df["fmov_str"]  = [x[1] for x in sorted_fmov_list]

    df["uc"]    = df["uc_str"].str.split("\n")
    df["fhr"]   = df["fhr_str"].str.split("\n")
    df["fmov"]  = df["fmov_str"].where(df["fmov_str"].astype(bool), None).str.split("\n")

    df["gest_age"] = df.apply(
        lambda r: extract_gest_age(r["conclusion"], r["basic_info"]),
        axis=1
    ).astype("Int64")

    df.rename(columns={"id": "_id"}, inplace=True)

    df.drop(
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

    records = df.to_dict(orient="records")

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