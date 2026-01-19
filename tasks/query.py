from utils.common import patch_job
from utils.query import df_to_records
from utils.filter import async_filter_record
from schemas.records import assert_records_match_schema
from database_manager.database.mongo import MongoDBConnector
from database_manager.database.mysql import SQLDBConnector
from database_manager.database.queries import RECRUITED

import anyio

from anyio import CapacityLimiter, create_task_group
from datetime import datetime


async def run_query_job(
    job_id: str, mongo: MongoDBConnector, sql: SQLDBConnector
) -> None:

    await patch_job(mongo, job_id, status="running", result=None, error=None)

    try:
        await query(mongo=mongo, sql=sql)
        await patch_job(mongo, job_id, status="succeeded", result=None, error=None)

    except Exception as e:
        await patch_job(mongo, job_id, status="failed", result=None, error=str(e))


async def query(mongo: MongoDBConnector, sql: SQLDBConnector) -> None:

    new_given_birth = await mongo.get_all_documents(
        coll_name="METADATA_REC",
        query={"processed": False},
        projection={"_id": 0, "mobile": 1},
    )

    new_given_birth_mobile = [i["mobile"] for i in new_given_birth]

    print(f"QUERIED FROM `METADATA_REC` ({len(new_given_birth_mobile)} PATIENTS)")

    if len(new_given_birth_mobile) == 0:
        print(f"NO NEW PATIENTS WHO HAVE GIVEN BIRTH")
    else:
        query_string = ",".join(new_given_birth_mobile)
        custom_query = RECRUITED.format(
            start="'2025-03-01 00:00:00'",
            end=f"'{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}'",
            numbers=query_string,
        )

        given_birth_df = await anyio.to_thread.run_sync(
            lambda: sql.query_to_dataframe(query=custom_query)
        )

        if len(given_birth_df) == 0:
            print(f"NO RECORDS FROM PATIENTS: {new_given_birth_mobile}")
        else:
            given_birth_df["origin"] = "REC"

            print(f"QUERIED FROM NAVICAT ({len(given_birth_df)} MEASUREMENTS)")

            records = await df_to_records(given_birth_df)

            assert_records_match_schema(records, record_type="RAW")

            await mongo.upsert_documents_hashed(
                coll_name="RECORDS_RAW", records=records
            )

            print(f"[A] UPSERTED TO 'RECORDS_RAW' ({len(records)} RECORDS)")

        for m in new_given_birth_mobile:
            await mongo.patch_document(
                coll_name="METADATA_REC",
                query={"mobile": m},
                set_fields={"processed": True},
            )

            print(f"PATIENT {m} MARKED AS PROCESSED")

    not_given_birth = await mongo.get_all_documents(
        coll_name="METADATA_PRED", projection={"_id": 0, "mobile": 1}
    )

    not_given_birth_mobile = [i["mobile"] for i in not_given_birth]

    print(f"QUERIED FROM `METADATA_PRED` ({len(not_given_birth_mobile)} PATIENTS)")

    if len(not_given_birth_mobile) == 0:
        print(f"NO PATIENTS WHO HAVE NOT GIVEN BIRTH")
    else:
        query_string = ",".join(not_given_birth_mobile)
        custom_query = RECRUITED.format(
            start="'2025-03-01 00:00:00'",
            end=f"'{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}'",
            numbers=query_string,
        )

        not_given_birth_df = await anyio.to_thread.run_sync(
            lambda: sql.query_to_dataframe(query=custom_query)
        )

        if len(not_given_birth_df) == 0:
            print(f"NO RECORDS FROM PATIENTS: {not_given_birth_mobile}")
        else:
            not_given_birth_df["origin"] = "REC"

            print(f"QUERIED FROM NAVICAT ({len(not_given_birth_df)} MEASUREMENTS)")

            records = await df_to_records(not_given_birth_df)

            limiter = CapacityLimiter(50)
            filtered = []

            async def run_and_collect(r):
                res = await async_filter_record(r, limiter)
                if res is not None:
                    filtered.append(res)

            async with create_task_group() as tg:
                for r in records:
                    tg.start_soon(run_and_collect, r)

            assert_records_match_schema(filtered, record_type="FILT")

            await mongo.upsert_documents_hashed(
                coll_name="RECORDS_PRED", records=filtered
            )

            print(f"[A] UPSERTED TO 'RECORDS_PRED' ({len(filtered)} RECORDS)")
