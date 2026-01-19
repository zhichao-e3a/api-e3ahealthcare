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
    "age",
    "bmi",
    "had_pregnancy",
    "had_preterm",
    "had_surgery",
    "gdm",
    "pih",
    "delivery_type",
    "add",
    "onset",
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
            raise AssertionError(f"Row {i}: gest_age must be int")

        # origin: str
        if not isinstance(r["origin"], str):
            raise AssertionError(f"Row {i}: sql_utime must be str")

        # age: int | None
        if not (isinstance(r["age"], int) or r["age"] is None):
            raise AssertionError(f"Row {i}: age must be int or None")

        # bmi: int | None
        if not (isinstance(r["bmi"], int) or r["bmi"] is None):
            raise AssertionError(f"Row {i}: bmi must be int or None")

        # had_pregnancy: int
        if not isinstance(r["had_pregnancy"], int):
            raise AssertionError(f"Row {i}: had_pregnancy must be int")

        # had_preterm: int
        if not isinstance(r["had_preterm"], int):
            raise AssertionError(f"Row {i}: had_preterm must be int")

        # had_surgery: int
        if not isinstance(r["had_surgery"], int):
            raise AssertionError(f"Row {i}: had_surgery must be int")

        # gdm: int
        if not isinstance(r["gdm"], int):
            raise AssertionError(f"Row {i}: gdm must be int")

        # pih: int
        if not isinstance(r["pih"], int):
            raise AssertionError(f"Row {i}: pih must be int")

        # delivery_type: str | None
        if not (isinstance(r["delivery_type"], str) or r["delivery_type"] is None):
            raise AssertionError(f"Row {i}: delivery_type must be str or None")

        # add: str | None
        if not (isinstance(r["add"], str) or r["add"] is None):
            raise AssertionError(f"Row {i}: add must be str or None")

        # onset: str | None
        if not (isinstance(r["onset"], str) or r["onset"] is None):
            raise AssertionError(f"Row {i}: onset must be str or None")
