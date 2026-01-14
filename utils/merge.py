import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo

from scipy.signal import find_peaks, welch, butter, sosfiltfilt

FS = 1.0

def moving_median(x: np.ndarray, k: int) -> np.ndarray:

    k = int(k) if int(k) % 2 == 1 else int(k) + 1
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    out = np.empty_like(x, dtype=float)

    for i in range(len(x)):
        w = xp[i:i+k]
        out[i] = np.nanmedian(w)

    return out

def linear_fill_short_gaps(x, max_len=3):

    y = x.copy().astype(float)
    n = len(y)
    i = 0

    while i < n:
        if np.isnan(y[i]):
            j = i
            while j < n and np.isnan(y[j]): j += 1
            run = j - i
            if run <= max_len:
                x0 = y[i-1] if i > 0 and np.isfinite(y[i-1]) else y[j]
                x1 = y[j]   if j < n and np.isfinite(y[j])   else y[i-1]
                y[i:j] = np.linspace(x0, x1, run+2)[1:-1]
            i = j
        else:
            i += 1

    return y

def sliding_windows(n, w=120, s=30):

    start = 0

    while start + w <= n:
        yield start, start + w
        start += s

def _safe_fill_numeric(x: np.ndarray, fallback: float = 0.0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    if not np.isfinite(med):
        med = fallback
    return np.nan_to_num(x, nan=med, posinf=med, neginf=med)

def butter_filter(
        x: np.ndarray, cutoff_hz, btype: str, fs: float = FS, order: int = 4, fallback: str = "return_filled"
) -> np.ndarray:

    x = np.asarray(x, dtype=float)
    x_filled = _safe_fill_numeric(x, fallback=0.0)

    nyq = fs * 0.5
    if isinstance(cutoff_hz, (tuple, list, np.ndarray)):
        if len(cutoff_hz) != 2:
            raise ValueError("Band cutoff must be (low, high).")
        low, high = float(cutoff_hz[0]), float(cutoff_hz[1])
        if not (0 < low < high < nyq):
            raise ValueError(f"Band edges must satisfy 0<low<high<{nyq} Hz. Got {cutoff_hz}.")
        Wn = (low, high)
    else:
        c = float(cutoff_hz)
        if not (0 < c < nyq):
            raise ValueError(f"Cutoff must be 0<{c}<{nyq} Hz.")
        Wn = c

    sos = butter(order, Wn, btype=btype, fs=fs, output="sos")

    padlen = 3 * order
    n = x_filled.size
    if n <= padlen:
        if fallback == "median":
            med = float(np.nanmedian(x)) if np.isfinite(np.nanmedian(x)) else 0.0
            return np.full_like(x_filled, med)
        else:
            return x_filled

    y = sosfiltfilt(sos, x_filled, padtype="odd", padlen=padlen)

    y = _safe_fill_numeric(y, fallback=float(np.nanmedian(y)))

    return y

def pad_signal(x: np.ndarray, target_len: int = 2048) -> np.ndarray:

    x = np.asarray(x, dtype=float)
    x = np.nan_to_num(x, nan=np.nanmedian(x) if np.isfinite(np.nanmedian(x)) else 0.0)

    if len(x) == target_len:
        return x

    else:
        pad_val = np.nanmedian(x)
        pad_left = (target_len - len(x)) // 2
        pad_right = target_len - len(x) - pad_left
        x_padded = np.pad(x, (pad_left, pad_right), mode="constant", constant_values=pad_val)

        return x_padded

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

def hampel_filter_1d(x, k=11, n_sigmas=3.0):

    x = x.astype(float).copy()
    k = int(k) if k % 2 == 1 else int(k) + 1
    half = k // 2
    xp = np.pad(x, (half, half), mode="edge")
    med = np.empty_like(x, dtype=float)
    mad = np.empty_like(x, dtype=float)

    for i in range(len(x)):
        w = xp[i:i+k]
        m = np.median(w)
        med[i] = m
        mad[i] = np.median(np.abs(w - m)) + 1e-9

    thresh = n_sigmas * 1.4826 * mad
    mask = np.abs(x - med) > thresh
    y = x.copy()
    y[mask] = med[mask]

    return y, mask

def uc_preprocess(
        uc: np.ndarray, lpf_cutoff_hz: float = 0.1, median_k: int = 11, baseline_k: int = 61, hampel_k: int = 11, hampel_sigmas: float = 3.0,
):

    # Input guards
    x = np.asarray(uc, float).copy()
    median_k = median_k if median_k % 2 == 1 else median_k + 1
    baseline_k = baseline_k if baseline_k % 2 == 1 else baseline_k + 1

    # Light denoise (Low-pass + Moving Median) -> Spike suppression -> Baseline subtraction
    x = butter_filter(x, lpf_cutoff_hz, "low")
    x = moving_median(x, median_k)
    x, m = hampel_filter_1d(x, k=hampel_k, n_sigmas=hampel_sigmas)
    base = moving_median(x, baseline_k)
    x = x - base

    # Uncomment to return masks
    # masks = {'spike_hampel' : m}

    return x

def uc_detect_contractions(uc_proc: np.ndarray, min_distance_s: int = 100, prom_scale_mad: float = 1.0):

    x = uc_proc.astype(float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) + 1e-6
    prominence = prom_scale_mad * mad
    peaks, props = find_peaks(
        np.nan_to_num(x, nan=med),
        distance=min_distance_s,
        prominence=prominence
    )

    props["mad_used"] = mad
    props["prominence_used"] = prominence

    return peaks, props

def uc_time_features(win: np.ndarray, peaks: np.ndarray, props: dict) -> dict:

    feats = {}

    if peaks.size == 0:

        feats.update(
            {
                "uc_n_contr": 0,
                "uc_amp_mean": 0.0,
                "uc_dur_mean": 0.0,
                "uc_ici_mean": 0.0,
                "uc_duty_cycle": 0.0,
                "uc_rise_slope_mean": 0.0,
                "uc_fall_slope_mean": 0.0,
                "uc_area_mean": 0.0,
                "uc_amp_sd": 0.0,
                "uc_dur_sd": 0.0,
                "uc_ici_sd": 0.0,
            }
        )

        return feats

    amp = props.get("prominences", np.array([])).astype(float)
    left = props.get("left_bases", np.array([])).astype(int)
    right = props.get("right_bases", np.array([])).astype(int)

    n = len(peaks)
    dur = (right - left).astype(float)
    ici = np.diff(peaks).astype(float) if n > 1 else np.array([])

    duty = float(np.sum(dur) / len(win))

    rises, falls, areas = [], [], []
    x = np.nan_to_num(win, nan=np.nanmedian(win))
    for pk, l, r in zip(peaks, left, right):

        l = int(l); r = int(r)

        if r <= l or pk <= l or pk >= r:
            continue

        rises.append((x[pk] - x[l]) / max(1, (pk - l)))
        falls.append((x[r] - x[pk]) / max(1, (r - pk)))
        areas.append(float(np.trapezoid(x[l:r] - x[l], dx=1.0)))

    feats.update(
        {
            "uc_n_contr": int(n),
            "uc_amp_mean": float(np.mean(amp)) if amp.size else 0.0,
            "uc_dur_mean": float(np.mean(dur)) if dur.size else 0.0,
            "uc_ici_mean": float(np.mean(ici)) if ici.size else 0.0,
            "uc_duty_cycle": duty,
            "uc_rise_slope_mean": float(np.mean(rises)) if rises else 0.0,
            "uc_fall_slope_mean": float(np.mean(falls)) if falls else 0.0,
            "uc_area_mean": float(np.mean(areas)) if areas else 0.0,
            "uc_amp_sd": float(np.std(amp, ddof=1)) if amp.size > 1 else 0.0,
            "uc_dur_sd": float(np.std(dur, ddof=1)) if dur.size > 1 else 0.0,
            "uc_ici_sd": float(np.std(ici, ddof=1)) if ici.size > 1 else 0.0,
        }
    )

    return feats

def uc_spectral_features(win: np.ndarray) -> dict:

    x = np.nan_to_num(win, nan=np.nanmedian(win))
    n = len(x)
    nperseg = min(256, n)  # cap at window length
    f, pxx = welch(x, fs=FS, window="hamming", nperseg=nperseg)  # no noverlap arg
    mask_low = (f >= 0.003) & (f <= 0.02)
    dom_freq = float(f[mask_low][np.argmax(pxx[mask_low])]) if np.any(mask_low) else 0.0
    low_power = float(np.trapezoid(pxx[f < 0.03], f[f < 0.03]))
    total_power = float(np.trapezoid(pxx[f <= 0.1], f[f <= 0.1]))
    p = pxx / (pxx.sum() + 1e-12)

    return {
        "uc_dom_freq": dom_freq,
        "uc_power_sub003": low_power,
        "uc_total_power": total_power,
        "uc_spec_entropy": float(np.sum(-p * np.log2(p + 1e-12))),
    }

def extract_uc_features(
        uc_proc: np.ndarray, window_s: int = 120, stride_s: int = 30, coverage_thresh: float = 0.8, min_distance_s: int = 100, prom_scale_mad: float = 1.0
):

    rows = [] ; n = len(uc_proc)
    for s, e in sliding_windows(n, w=window_s, s=stride_s):

        win = uc_proc[s:e]
        cov = float(np.isfinite(win).mean())

        if cov < coverage_thresh:
            continue

        med = np.nanmedian(win)
        mad = np.nanmedian(np.abs(win - med)) + 1e-6
        prominence = prom_scale_mad * mad
        peaks, props = find_peaks(
            np.nan_to_num(win, nan=med),
            distance=min_distance_s,
            prominence=prominence
        )

        feats = {
            "t_start": int(s),
            "t_end": int(e),
            "coverage": cov,
            "uc_mad_local": float(mad),
            "uc_prominence_used": float(prominence),
        }

        feats.update(uc_time_features(win, peaks, props))
        feats.update(uc_spectral_features(win))

        rows.append(feats)

    return rows

def _runs(mask: np.ndarray) -> list[tuple[int,int]]:

    if mask.size == 0:
        return []

    m = np.concatenate([[False], mask, [False]])
    idx = np.flatnonzero(m[1:] != m[:-1])

    return list(zip(idx[0::2], idx[1::2]))

def clamp_out_of_range(fhr: np.ndarray, lo: float = 50.0, hi: float = 210.0) -> tuple[np.ndarray, np.ndarray]:

    x = fhr.astype(float).copy()
    bad = (x < lo) | (x > hi)
    x[bad] = np.nan

    return x, bad

def mask_spikes(fhr: np.ndarray, window: int = 11, threshold_bpm: float = 25.0) -> tuple[np.ndarray, np.ndarray]:

    x = fhr.astype(float).copy()
    med = moving_median(x, window)
    spike = np.abs(x - med) > threshold_bpm
    x[spike] = np.nan

    return x, spike

def dropout_detection(fhr: np.ndarray, max_flat_seconds: int = 3, atol: float = 0.1):

    x = fhr.astype(float)

    same_as_prev = np.zeros_like(x, dtype=bool)
    same_as_prev[1:] = np.isfinite(x[1:]) & np.isfinite(x[:-1]) & (np.abs(x[1:] - x[:-1]) <= atol)

    flat_mask = np.zeros_like(x, dtype=bool)
    run_len = 1

    for i in range(1, len(x)):
        if same_as_prev[i]:
            run_len += 1
        else:
            if run_len >= max_flat_seconds:
                flat_mask[i-run_len:i] = True
            run_len = 1

    if run_len >= max_flat_seconds:
        flat_mask[len(x)-run_len:len(x)] = True

    segments = _runs(flat_mask)

    return flat_mask, segments

def fhr_preprocess(
        fhr: np.ndarray, lo: float = 50.0, hi: float = 210.0, spike_window: int = 11, spike_thresh_bpm: float = 25.0, flat_seconds: int = 3, flat_atol: float = 0.1,
):

    # Clamp -> Spike Mask -> Dropout Detection
    x1, m_out = clamp_out_of_range(fhr, lo, hi)
    x2, m_spk = mask_spikes(x1, window=spike_window, threshold_bpm=spike_thresh_bpm)
    m_flat, flat_segments = dropout_detection(x2, max_flat_seconds=flat_seconds, atol=flat_atol)

    x_out = x2.copy()
    x_out[m_flat] = np.nan

    # Uncomment to return masks
    # masks = {
    #     "out_of_range": m_out,
    #     "spikes": m_spk,
    #     "flatline": m_flat,
    #     "flat_segments": flat_segments,
    #     "artifact_mask_any": m_out | m_spk | m_flat,
    # }

    return x_out

def coverage_pct(x: np.ndarray) -> float:

    if x.size == 0:
        return 0.0

    return float(np.isfinite(x).mean() * 100.0)

def make_tracks(fhr_with_nans):

    raw_light = linear_fill_short_gaps(fhr_with_nans, max_len=3)

    x = np.copy(raw_light)
    x[~np.isfinite(x)] = np.nanmedian(x)

    filtered = butter_filter(x, 0.01, "high")
    filtered = butter_filter(filtered, 0.3, "low")

    return raw_light, filtered

def fhr_accel_decel_features(x, accel_thr=15.0, decel_thr=-15.0, min_len=15):

    k = 31
    pad = k//2
    xp = np.pad(x, (pad, pad), mode="edge")
    baseline = np.array([np.median(xp[i:i+k]) for i in range(len(x))])

    d = x - baseline
    a_mask = d >= accel_thr
    d_mask = d <= decel_thr

    def segments(mask):
        m = np.concatenate([[False], mask, [False]])
        idx = np.flatnonzero(m[1:] != m[:-1])
        return [(s, e) for s, e in zip(idx[0::2], idx[1::2]) if (e - s) >= min_len]

    acc = segments(a_mask)
    dec = segments(d_mask)

    def summarize(segs):
        if not segs:
            return dict(count=0, amp_mean=0.0, dur_mean=0.0, auc_mean=0.0)
        amps, durs, aucs = [], [], []
        for s, e in segs:
            seg = d[s:e]
            amps.append(np.max(seg))
            durs.append(e - s)
            aucs.append(np.trapezoid(seg, dx=1.0))
        return dict(count=len(segs),
                    amp_mean=float(np.mean(amps)),
                    dur_mean=float(np.mean(durs)),
                    auc_mean=float(np.mean(aucs)))

    acc_d = summarize(acc)
    dec_d = summarize(dec)

    return {
        "fhr_accel_count"       : acc_d["count"],
        "fhr_accel_amp_mean"    : acc_d["amp_mean"],
        "fhr_accel_dur_mean"    : acc_d["dur_mean"],
        "fhr_accel_auc_mean"    : acc_d["auc_mean"],
        "fhr_decel_count"       : dec_d["count"],
        "fhr_decel_amp_mean"    : dec_d["amp_mean"],
        "fhr_decel_dur_mean"    : dec_d["dur_mean"],
        "fhr_decel_auc_mean"    : dec_d["auc_mean"],
    }

def fhr_welch_features(x: np.ndarray, fs: float = FS):

    x = np.asarray(x, dtype=float)
    if not np.isfinite(x).any():
        return {
            "psd_total": np.nan, "psd_vlf": np.nan, "psd_lf": np.nan, "psd_hf": np.nan,
            "ratio_lf_hf": np.nan, "ratio_vlf_total": np.nan
        }

    med = np.nanmedian(x) if np.isfinite(np.nanmedian(x)) else 0.0
    x = np.nan_to_num(x, nan=med, posinf=med, neginf=med)

    nperseg = min(256, len(x))
    if nperseg < 64:

        return {
            "psd_total": np.nan, "psd_vlf": np.nan, "psd_lf": np.nan, "psd_hf": np.nan,
            "ratio_lf_hf": np.nan, "ratio_vlf_total": np.nan
        }

    noverlap = nperseg // 2
    if noverlap >= nperseg:
        noverlap = nperseg - 1

    f, pxx = welch(
        x, fs=fs, window="hamming",
        nperseg=nperseg, noverlap=noverlap,
        detrend="constant", scaling="density", average="mean"
    )

    bands = {"vlf": (0.00, 0.03), "lf": (0.03, 0.15), "hf": (0.15, 0.50)}

    def band_power(lo, hi):
        idx = (f >= lo) & (f < hi)
        if not np.any(idx): return 0.0
        return np.trapezoid(pxx[idx], f[idx])

    vlf = band_power(*bands["vlf"])
    lf  = band_power(*bands["lf"])
    hf  = band_power(*bands["hf"])

    total = vlf + lf + hf

    return {
        "psd_total": total,
        "psd_vlf": vlf, "psd_lf": lf, "psd_hf": hf,
        "ratio_lf_hf": (lf / hf) if hf > 0 else np.nan,
        "ratio_vlf_total": (vlf / total) if total > 0 else np.nan,
    }

def fhr_time_features(x):

    v = x[np.isfinite(x)]

    if v.size < 2:
        return dict(fhr_mean=np.nan, fhr_median=np.nan, fhr_sdnn=np.nan,
                    fhr_rmssd=np.nan, fhr_range=np.nan, fhr_skew=np.nan, fhr_kurt=np.nan)

    dif = np.diff(v)
    mean = float(np.mean(v))
    sd = float(np.std(v, ddof=1))
    rmssd = float(np.sqrt(np.mean(dif**2))) if dif.size else 0.0
    r = float(np.max(v) - np.min(v))
    skew = float(((v-mean)**3).mean() / (sd**3 + 1e-12)) if sd>0 else 0.0
    kurt = float(((v-mean)**4).mean() / (sd**4 + 1e-12)) if sd>0 else 0.0

    return dict(fhr_mean=mean, fhr_median=float(np.median(v)),
                fhr_sdnn=sd, fhr_rmssd=rmssd, fhr_range=r,
                fhr_skew=skew, fhr_kurt=kurt)

def extract_fhr_features(
        fhr_clean: np.ndarray, window_s: int = 120, stride_s: int = 30, coverage_thresh: float = 0.8, accel_thr: float = 15.0, decel_thr: float = -15.0,
):

    raw_light, filtered = make_tracks(fhr_clean)
    rows = [] ; n = len(fhr_clean)

    for s, e in sliding_windows(n, w=window_s, s=stride_s):

        win_raw = raw_light[s:e]
        win_flt = filtered[s:e]
        cov = float(np.isfinite(win_raw).mean())

        if cov < coverage_thresh:
            continue

        feats = dict(t_start=int(s), t_end=int(e), coverage=cov)
        feats.update(fhr_time_features(win_raw))
        feats.update(fhr_accel_decel_features(win_raw, accel_thr=accel_thr, decel_thr=decel_thr))
        feats.update(fhr_welch_features(win_flt))

        rows.append(feats)

    return rows

def process_row(row):

    # 2 fields
    m_datetime = datetime.strptime(row['measurement_date'], '%Y-%m-%d %H:%M:%S')
    b_datetime = datetime.strptime(row['add'], '%Y-%m-%d %H:%M') if row['add'] else m_datetime

    target = (b_datetime-m_datetime).total_seconds()/60/60/24
    if target < 0 or target > 100:
        return None

    ga_exit = row['gest_age'] + target

    # 8 fields
    static_data = [
        # row['age'] if (row['age'] is not None and pd.notna(row['age'])) else 0,
        # row['bmi'] if (row['bmi'] is not None and pd.notna(row['bmi'])) else 0,
        row['age'],
        row['bmi'],
        row['had_pregnancy'],
        row['had_preterm'],
        row['had_surgery'],
        row['gdm'],
        row['pih'],
        row['gest_age']
    ]

    # 2 fields

    if len(row['uc']) > 2048:
        return None

    uc_padded   = pad_signal(np.array(row['uc']).astype(float))
    fhr_padded  = pad_signal(np.array(row['fhr']).astype(float))
    fmov_padded = pad_signal(np.array(row['fmov']).astype(float)) if row['fmov'] is not None else None

    uc_windows  = extract_uc_features(uc_preprocess(uc_padded))
    fhr_windows = extract_fhr_features(fhr_preprocess(fhr_padded))

    record = {
        '_id'               : f"{row['mobile']}_{row['measurement_date']}",
        'mobile'            : row['mobile'],
        'measurement_date'  : row['measurement_date'],
        'static'            : static_data,
        'uc_raw'            : row['uc'],
        'fhr_raw'           : row['fhr'],
        'fmov_raw'          : row['fmov'],
        'uc_padded'         : uc_padded.tolist(),
        'fhr_padded'        : fhr_padded.tolist(),
        'fmov_padded'       : fmov_padded.tolist() if fmov_padded is not None else None,
        'uc_windows'        : uc_windows,
        'fhr_windows'       : fhr_windows,
        'add'               : row['add'],
        'target'            : target,
        'ga_exit'           : ga_exit,
        'preterm'           : 1 if ga_exit//7 < 37 else 0
    }

    return record