from pydantic import BaseModel
from typing import Any, Dict, Literal, Optional

class JobOut(BaseModel):

    job_id      : str
    type        : str
    status      : str
    created_at  : str
    updated_at  : str
    result      : Optional[Dict[str, Any]] = None
    error       : Optional[str] = None