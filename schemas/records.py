from __future__ import annotations
from typing import Any, Dict, Sequence

REQUIRED = {
    "_id",
    "mobile",
    "start_test_ts",
    "measurement_date",
    "uc",
    "fhr",
    "fmov",
    "gest_age",
    "origin",
    "sql_utime",
}

ALLOWED = set(REQUIRED)


def assert_records_match_schema(
    records: Sequence[Dict[str, Any]], record_type: str
) -> None:

    for i, r in enumerate(records):

        if not isinstance(r, Dict):
            raise AssertionError(f"Row {i}: record is not a dict")

        keys = set(r.keys())

        missing = REQUIRED - keys
        if missing:
            raise AssertionError(f"Row {i}: missing keys {sorted(missing)}")

        if record_type == "FILT":
            ALLOWED.update({"doc_hash", "ctime", "utime"})
        extra = keys - ALLOWED
        if extra:
            raise AssertionError(f"Row {i}: unexpected keys {sorted(extra)}")

        # _id: int
        if not isinstance(r["_id"], int):
            raise AssertionError(f"Row {i}: _id must be int")

        # mobile: str
        if not isinstance(r["mobile"], str):
            raise AssertionError(f"Row {i}: mobile must be str")

        # start_test_ts: str
        if not isinstance(r["start_test_ts"], str):
            raise AssertionError(f"Row {i}: start_test_ts must be str or None")

        # measurement_date: str
        if not isinstance(r["measurement_date"], str):
            raise AssertionError(f"Row {i}: measurement_date must be str")

        # uc: List[str]
        if not isinstance(r["uc"], list) and all(isinstance(x, str) for x in r["uc"]):
            raise AssertionError(f"Row {i}: uc must be list[str]")

        # fhr: List[str]
        if not isinstance(r["fhr"], list) and all(isinstance(x, str) for x in r["fhr"]):
            raise AssertionError(f"Row {i}: fhr must be list[str]")

        # fmov: List[str] | None
        if not (
            (isinstance(r["fmov"], list) and all(isinstance(x, str) for x in r["fmov"]))
            or r["fmov"] is None
        ):
            raise AssertionError(f"Row {i}: fmov must be list[str] or None")

        # gest_age: int | None
        if not isinstance(r["gest_age"], int):
            if record_type == "RAW":
                if not r["gest_age"] is None:
                    raise AssertionError(f"Row {i}: gest_age must be int or None")
            if record_type == "FILT":
                raise AssertionError(f"Row {i}: gest_age must be int")

        # origin: str
        if not isinstance(r["origin"], str):
            raise AssertionError(f"Row {i}: sql_utime must be str")

        # sql_utime: str
        if not isinstance(r["sql_utime"], str):
            raise AssertionError(f"Row {i}: sql_utime must be str")
