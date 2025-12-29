import anyio

from datetime import datetime
from zoneinfo import ZoneInfo

def extract_fetal_movement(raw_fmov, start_ts):

    fmov_idx = [] ; unique_time_set = set()
    start_dt = datetime.strptime(start_ts, '%Y-%m-%d %H:%M:%S')
    start_dt = start_dt.replace(tzinfo=ZoneInfo("Asia/Singapore"))

    for _fmov in raw_fmov:
        fmov_unix   = _fmov.split('：')[1].split(' ')[0]
        fmov_deg    = _fmov.split('：')[2]
        fmov_dt     = datetime.fromtimestamp(int(fmov_unix), tz=ZoneInfo("Asia/Singapore"))
        if fmov_dt < start_dt:
            continue
        idx         = fmov_dt-start_dt
        idx_s       = idx.seconds
        fmov_tuple  = (idx_s, fmov_deg)

        if idx_s not in unique_time_set:
            fmov_idx.append(fmov_tuple)
            unique_time_set.add(idx_s)

    fmov_idx.sort(key=lambda x: x[0])
    last = fmov_idx[-1][0]

    record = ["0" for _ in range(last)]

    for fm in fmov_idx:
        record[fm[0]-1] = fm[1]

    return record

def filter_record(record):

    gest_age    = record['gest_age']
    uc_data     = record['uc']
    fhr_data    = record['fhr']

    if gest_age is None or len(uc_data) < 60 * 20 or len(fhr_data) < 60 * 20:
        return None

    max_len = max(len(uc_data), len(fhr_data))
    d_uc = max_len - len(uc_data);
    d_fhr = max_len - len(fhr_data)
    if d_uc > 0:
        uc_data.extend(['0'] * d_uc)
    if d_fhr > 0:
        fhr_data.extend(['0'] * d_fhr)

    fmov_data = extract_fetal_movement(record['fmov'], record['measurement_date'])\
        if record['fmov'] else None

    if fmov_data is not None:
        if len(fmov_data) < max_len:
            d_fmov = max_len - len(fmov_data)
            fmov_data.extend(['0'] * d_fmov)
        elif len(fmov_data) > max_len:
            extra = len(fmov_data) - max_len
            uc_data.extend(["0"] * extra)
            fhr_data.extend(["0"] * extra)

    record['uc'] = uc_data
    record['fhr'] = fhr_data
    record['fmov'] = fmov_data

    return record

async def async_filter_record(record, limiter):

    async with limiter:
        return await anyio.to_thread.run_sync(filter_record, record)