from utils.common import patch_job
from utils.merge import process_row
from database_manager.database.mongo import MongoDBConnector

import os
import json
import pandas as pd

from tqdm.auto import tqdm
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

async def run_merge_job(job_id: str, mongo: MongoDBConnector) -> None:

    await patch_job(mongo, job_id, status="running", result=None, error=None)

    try:
        await merge(mongo=mongo)
        await patch_job(mongo, job_id, status="succeeded", result=None, error=None)

    except Exception as e:
        await patch_job(mongo, job_id, status="failed", result=None, error=str(e))

async def merge(mongo: MongoDBConnector) -> None:

    meta_pred    = await mongo.get_all_documents(coll_name="METADATA_PRED")
    meta_pred_df = pd.DataFrame(meta_pred)
    print(f"[PRED] {len(meta_pred_df)} metadata records fetched")

    measurements_pred    = await mongo.get_all_documents(coll_name="RECORDS_PRED")
    measurements_pred_df = pd.DataFrame(measurements_pred)
    print(f"[PRED] {len(measurements_pred_df)} measurement records fetched")

    merged_pred_df = measurements_pred_df.merge(meta_pred_df, how="left", on='mobile')
    print(f"[PRED] {len(merged_pred_df)} merged records")

    merged_pred = merged_pred_df.to_dict(orient="records")

    dataset = []
    with ProcessPoolExecutor(max_workers=max(1, os.cpu_count() - 1)) as executor:

        futures = [executor.submit(process_row, row) for row in merged_pred]

        for fut in tqdm(as_completed(futures), total=len(futures)):

            rec = fut.result()
            if rec is not None:
                dataset.append(rec)

    pred_out = f"/app/datasets/{datetime.now().strftime('%Y%m%d')}_dataset_pred.json"
    with open(pred_out, "w") as f:
        f.write(json.dumps(dataset))

    print(len(dataset), "measurements written to", pred_out)

    meta_rec    = await mongo.get_all_documents(coll_name="METADATA_REC")
    meta_hist   = await mongo.get_all_documents(coll_name="METADATA_HIST")
    meta_all    = meta_rec + meta_hist
    meta_all_df = pd.DataFrame(meta_all)
    print(f"[ALL] {len(meta_all_df)} metadata records fetched")

    measurements_all    = await mongo.get_all_documents(coll_name="RECORDS_FILT")
    measurements_all_df = pd.DataFrame(measurements_all)
    print(f"[ALL] {len(measurements_all_df)} measurement records fetched")

    merged_all_df = measurements_all_df.merge(meta_all_df, how="left", on='mobile')
    print(f"[ALL] {len(merged_all_df)} merged records")

    merged_all = merged_all_df.to_dict(orient="records")

    dataset = []
    with ProcessPoolExecutor(max_workers=max(1, os.cpu_count() - 1)) as executor:

        futures = [executor.submit(process_row, row) for row in merged_all]

        for fut in tqdm(as_completed(futures), total=len(futures)):

            rec = fut.result()
            if rec is not None:
                dataset.append(rec)

    all_out = f"/app/datasets/{datetime.now().strftime('%Y%m%d')}_dataset_all.json"
    with open(all_out, "w") as f:
        f.write(json.dumps(dataset))