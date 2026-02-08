from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel

from ..api import DBCaseConfig, DBConfig


class LSMVecConfig(DBConfig):
    """
    DB level configs for LSM-Vec.

    Notes:
    - LSM-Vec is embedded (local), so we use filesystem paths instead of uri/host.
    - Avoid empty string defaults due to DBConfig validators.
    """

    db_root: str = "./run/lsmvec"
    vector_file_name: str = "vector.log"

    # Always remove existing db dir when drop_old is True in client.
    # You can also force clean start even when drop_old is False.
    always_clean_start: bool = True

    # Vector storage related configs (best effort, set only if binding exposes them)
    vec_storage: int = 0  # 0 BasicVectorStorage, 1 PagedVectorStorage
    paged_cache_pages: int = 256

    def to_dict(self) -> dict:
        return self.dict()


class LSMVecIndexConfig(BaseModel, DBCaseConfig):
    """
    Case specific configs for LSM-Vec index and search.

    The binding fields are best effort. We will set them only if attributes exist.
    """
    metric_type: Optional[MetricType] = None

    # Graph params
    M: int = 8
    Mmax: int = 16
    Ml: int = 1
    efc: int = 64  # efConstruction

    # Search params (best effort)
    ef_search: Optional[int] = None

    def index_param(self) -> Dict[str, Any]:
        return {
            "M": self.M,
            "Mmax": self.Mmax,
            "Ml": self.Ml,
            "efc": self.efc,
        }

    def search_param(self) -> Dict[str, Any]:
        if self.ef_search is None:
            return {}
        return {"ef_search": self.ef_search}
