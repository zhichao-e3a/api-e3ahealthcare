from utils.common import patch_job
from utils.merge import process_row
from database_manager.database.mongo import MongoDBConnector

import os
import json

from tqdm.auto import tqdm
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from typing import Optional


async def run_merge_job(job_id: str, mongo: MongoDBConnector) -> None:

    await patch_job(mongo, job_id, status="running", result=None, error=None)

    try:
        await merge(mongo=mongo)
        await patch_job(mongo, job_id, status="succeeded", result=None, error=None)

    except Exception as e:
        await patch_job(mongo, job_id, status="failed", result=None, error=str(e))


def _max_workers() -> int:

    env_val = os.getenv("MERGE_MAX_WORKERS")
    if env_val and env_val.isdigit():
        return max(1, int(env_val))

    cpu_count = os.cpu_count() or 1
    return max(1, cpu_count - 1)


def _build_meta_index(meta_rows):

    index = {}
    for row in meta_rows:
        mobile = row.get("mobile")
        if mobile is not None:
            index[mobile] = row
    return index


def _build_meta_index_preferring_rec(meta_rows, preferred_origin: str):

    index = {}
    for row in meta_rows:
        mobile = row.get("mobile")
        if mobile is None:
            continue
        if mobile not in index:
            index[mobile] = row
            continue
        print(f"{mobile} HAS REPEATED ENTRY IN METADATA_HIST & METADATA_REC")
        current = index[mobile]
        if (
            current.get("origin") != preferred_origin
            and row.get("origin") == preferred_origin
        ):
            index[mobile] = row
    return index


def _iter_merged_rows(measurements, meta_index):

    for row in measurements:
        mobile = row.get("mobile")
        if mobile is None:
            continue
        meta = meta_index.get(mobile)
        if meta:
            merged = row.copy()
            merged.update(meta)
            yield merged


def _write_json_list(path, records_iter, total: Optional[int] = None):

    count = 0
    with open(path, "w") as f:
        f.write("[")
        first = True
        for rec in tqdm(records_iter, total=total):
            if rec is None:
                continue
            if not first:
                f.write(",")
            f.write(json.dumps(rec))
            first = False
            count += 1
        f.write("]")

    return count


async def merge(mongo: MongoDBConnector) -> None:

    meta_pred = await mongo.get_all_documents(coll_name="METADATA_PRED")
    print(f"[PRED] {len(meta_pred)} metadata records fetched")

    measurements_pred = await mongo.get_all_documents(coll_name="RECORDS_PRED")
    print(f"[PRED] {len(measurements_pred)} measurement records fetched")

    meta_pred_index = _build_meta_index(meta_pred)
    merged_pred_iter = _iter_merged_rows(measurements_pred, meta_pred_index)
    print(f"[PRED] {len(measurements_pred)} merged records")

    pred_out = f"/app/datasets/{datetime.now().strftime('%Y%m%d')}_dataset_pred.json"
    with ProcessPoolExecutor(max_workers=_max_workers()) as executor:
        rec_iter = executor.map(process_row, merged_pred_iter, chunksize=25)
        written = _write_json_list(pred_out, rec_iter, total=len(measurements_pred))

    print(written, "measurements written to", pred_out)

    meta_rec = await mongo.get_all_documents(coll_name="METADATA_REC")
    meta_hist = await mongo.get_all_documents(coll_name="METADATA_HIST")
    meta_all = meta_rec + meta_hist
    print(f"[ALL] {len(meta_all)} metadata records fetched")

    measurements_all = await mongo.get_all_documents(coll_name="RECORDS_FILT")
    print(f"[ALL] {len(measurements_all)} measurement records fetched")

    meta_all_index = _build_meta_index_preferring_rec(meta_all, "REC")
    merged_all_iter = _iter_merged_rows(measurements_all, meta_all_index)
    print(f"[ALL] {len(measurements_all)} merged records")

    all_out = f"/app/datasets/{datetime.now().strftime('%Y%m%d')}_dataset_all.json"
    with ProcessPoolExecutor(max_workers=_max_workers()) as executor:
        rec_iter = executor.map(process_row, merged_all_iter, chunksize=25)
        written = _write_json_list(all_out, rec_iter, total=len(measurements_all))

    print(written, "measurements written to", all_out)
