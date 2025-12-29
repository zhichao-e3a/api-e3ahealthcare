from utils.common import patch_job
from utils.merge import bmi_choose_weight_kg

import pandas as pd

from database_manager.database.mongo import MongoDBConnector
from database_manager.database.mysql import SQLDBConnector
from database_manager.database.queries import HISTORICAL_METADATA_QUERY

async def run_merge_job(job_id: str, mongo: MongoDBConnector, sql: SQLDBConnector) -> None:

    await patch_job(mongo, job_id, status="running", result=None, error=None)

    try:
        await merge(mongo=mongo, sql=sql)
        await patch_job(mongo, job_id, status="succeeded", result=None, error=None)

    except Exception as e:
        await patch_job(mongo, job_id, status="failed", result=None, error=str(e))

async def merge(mongo: MongoDBConnector, sql: SQLDBConnector) -> None:

    surveyed_patients = await mongo.get_all_documents(
        coll_name="patients_unified",
        projection = {
            "_id"           : 0,
            "mobile"        : 1,
            "age"           : 1,
            "bmi"           : 1,
            "had_pregnancy" : 1,
            "had_preterm"   : 1,
            "had_surgery"   : 1,
            "gdm"           : 1,
            "pih"           : 1,
            "delivery_type" : 1,
            "add"           : 1,
            "onset"         : 1,
            "type"          : 1
        }
    )

    surveyed_patients_mobiles = set([i['mobile'] for i in surveyed_patients])
    surveyed_rec_patients_mobiles = set([i['mobile'] for i in surveyed_patients if i['type'] == 'rec'])
    surveyed_hist_patients_mobiles = set([i['mobile'] for i in surveyed_patients if i['type'] == 'hist'])

    print(f"QUERIED FROM `patients_unified` ({len(surveyed_patients_mobiles)} PATIENTS)")
    print(f"{len(surveyed_rec_patients_mobiles)} RECRUITED PATIENTS")
    print(f"{len(surveyed_hist_patients_mobiles)} HISTORICAL PATIENTS")
    print()

    measurements = await mongo.get_all_documents(
        coll_name=f"FILT_RECORDS",
        projection={
            "_id"               : 1,
            "mobile"            : 1,
            "start_test_ts"     : 1,
            "measurement_date"  : 1,
            "uc"                : 1,
            "fhr"               : 1,
            "fmov"              : 1,
            "gest_age"          : 1,
            "origin"            : 1
        }
    )

    all_patients_mobiles = set([i['mobile'] for i in measurements])
    recruited_patients_mobiles = set([i['mobile'] for i in measurements if i['origin'] == "REC"])
    historical_patients_mobiles = set([i['mobile'] for i in measurements if i['origin'] == "HIST"])

    print(f"QUERIED FROM `FILT_RECORDS` ({len(all_patients_mobiles)} PATIENTS)")
    print(f"{len(recruited_patients_mobiles)} RECRUITED PATIENTS")
    print(f"{len(historical_patients_mobiles)} HISTORICAL PATIENTS")
    print()

    print("INSIDE `patient_unified` BUT NO VALID MEASUREMENTS:")
    print(f"RECRUITED  : {surveyed_rec_patients_mobiles - recruited_patients_mobiles}")
    print(f"HISTORICAL : {surveyed_hist_patients_mobiles - historical_patients_mobiles}")
    print()

    print(f"QUERIED FROM `FILT_RECORDS` ({len(measurements)} MEASUREMENTS)")

    relevant_historical = historical_patients_mobiles - surveyed_hist_patients_mobiles
    query_str = ", ".join(list(relevant_historical))

    df = sql.query_to_dataframe(query=HISTORICAL_METADATA_QUERY.format(mobile_query_str=query_str))
    df_pivot = df.pivot(
        index=[i for i in df.columns if i not in ['record_type', 'record_answer']],
        columns='record_type',
        values='record_answer'
    ).reset_index()

    hist_metadata = df_pivot.copy()

    hist_metadata["age"] = pd.to_numeric(hist_metadata["age"], errors="coerce").astype("Int64")
    hist_metadata["bmi"] = hist_metadata.apply(
        lambda r: bmi_choose_weight_kg(height_cm=r["height"], weight_val=r["old_weight"]),
        axis=1
    )
    hist_metadata["had_pregnancy"] = (hist_metadata[1.0] > 1).astype(int)
    hist_metadata["had_preterm"]   = (hist_metadata[8.0] == 0).astype(int)
    hist_metadata["had_surgery"]   = (hist_metadata[13.0] == 0).astype(int)
    hist_metadata["gdm"]           = (hist_metadata[4.0] == 0).astype(int)
    hist_metadata["pih"]           = (hist_metadata[5.0] == 0).astype(int)
    hist_metadata["delivery_type"] = None
    hist_metadata["add"]           = (
        pd.to_datetime(hist_metadata["end_born_ts"], unit="s", utc=True)
        .dt.tz_convert("Asia/Singapore")
        .dt.strftime("%Y-%m-%d %H:%M")
    )
    hist_metadata["onset"]         = None
    hist_metadata["type"]          = "hist"

    cols = [
        "mobile", "age", "bmi",
        "had_pregnancy", "had_preterm", "had_surgery", "gdm", "pih",
        "delivery_type", "add", "onset", "type"
    ]

    hist_metadata = hist_metadata[cols]
    surveyed_metadata = pd.DataFrame(surveyed_patients)

    all_metadata = pd.concat([surveyed_metadata, hist_metadata], ignore_index=True)

    mobile_set = set()
    for _, i in all_metadata.iterrows():
        if i['mobile'] not in mobile_set:
            mobile_set.add(i['mobile'])
            continue
        print(f"REPEATED PATIENT: {i['mobile']}")

    print(f"QUERIED METADATA FOR {len(all_metadata)} PATIENTS")

    merged = pd.DataFrame(measurements).merge(all_metadata, how="left", on="mobile")

    print(f"COMPLETE MERGED DATA FOR {len(set(merged['mobile']))} PATIENTS")

    merged_docs = merged.to_dict(orient="records")

    await mongo.upsert_documents_hashed(
        coll_name="MERGED_RECORDS",
        records=merged_docs
    )

    print(f"UPSERTED TO `MERGED_RECORDS` ({len(merged_docs)} RECORDS)")