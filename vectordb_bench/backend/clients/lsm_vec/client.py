from __future__ import annotations

import logging
import os
import shutil
from contextlib import contextmanager
from typing import Iterable, Optional

from vectordb_bench.backend.filter import Filter, FilterOp

from ..api import DBCaseConfig, VectorDB
from .config import LSMVecConfig, LSMVecIndexConfig

log = logging.getLogger(__name__)


def _setAttrIfExists(obj: object, attrName: str, value: object) -> bool:
    if hasattr(obj, attrName):
        try:
            setattr(obj, attrName, value)
            return True
        except Exception as e:
            log.debug(f"Failed to set {attrName}={value}: {e}")
            return False
    return False


def _applyBestEffortIndexOptions(opts: object, caseConfig: Optional[LSMVecIndexConfig]) -> None:
    if caseConfig is None:
        return

    # The binding might use these names. We set only when exists.
    candidates = [
        ("M", caseConfig.M),
        ("Mmax", caseConfig.Mmax),
        ("Ml", caseConfig.Ml),
        ("efc", caseConfig.efc),
        ("efConstruction", caseConfig.efc),
        ("ef_construction", caseConfig.efc),
    ]
    for name, value in candidates:
        _setAttrIfExists(opts, name, value)


def _applyBestEffortStorageOptions(opts: object, dbConfig: LSMVecConfig) -> None:
    candidates = [
        ("vec_storage", dbConfig.vec_storage),
        ("vector_storage", dbConfig.vec_storage),
        ("paged_cache_pages", dbConfig.paged_cache_pages),
        ("pagedCachePages", dbConfig.paged_cache_pages),
    ]
    for name, value in candidates:
        _setAttrIfExists(opts, name, value)


def _makeSearchOptions(lsmVecModule: object, k: int, caseConfig: Optional[LSMVecIndexConfig]) -> object:
    searchOpts = lsmVecModule.SearchOptions()
    _setAttrIfExists(searchOpts, "k", k)

    if caseConfig is not None and caseConfig.ef_search is not None:
        for name in ["ef_search", "efSearch", "ef"]:
            if _setAttrIfExists(searchOpts, name, caseConfig.ef_search):
                break

    return searchOpts


class LsmVec(VectorDB):
    """
    VectorDBBench client for LSM-Vec (embedded local engine).

    Lifecycle:
    - __init__: store configs only (must be picklable across processes)
    - init: open DB and keep handle in self.db_
    - insert_embeddings: insert vectors
    - optimize: best effort flush or finalize
    - search_embedding: query knn
    """

    supported_filter_types: list[FilterOp] = [FilterOp.NonFilter]

    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: DBCaseConfig | None,
        collection_name: str | None = None,
        drop_old: bool = False,
        name: str = "LsmVec",
        **kwargs,
    ) -> None:
        self.name = name
        self.dim_ = int(dim)

        self.db_config_ = LSMVecConfig(**db_config)

        if db_case_config is None:
            self.case_config_ = None
        elif isinstance(db_case_config, LSMVecIndexConfig):
            self.case_config_ = db_case_config
        elif isinstance(db_case_config, dict):
            self.case_config_ = LSMVecIndexConfig(**db_case_config)
        else:
            # Fallback: try to parse from pydantic model dict
            self.case_config_ = LSMVecIndexConfig(**getattr(db_case_config, "dict", lambda: {})())

        if collection_name is None or collection_name == "":
            inferred = getattr(self.db_config_, "db_label", None)
            if not inferred:
                inferred = db_config.get("db_label") if isinstance(db_config, dict) else None
            collection_name = inferred or "VDBBench"

        safe = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in str(collection_name))
        self.collection_name_ = safe

        self.db_dir_ = os.path.join(self.db_config_.db_root, self.collection_name_)
        self.vector_file_path_ = os.path.join(self.db_dir_, self.db_config_.vector_file_name)

        self.db_ = None
        self.expr_ = ""

        self.need_clean_ = bool(drop_old) or bool(self.db_config_.always_clean_start)
        self.cleaned_ = False

    def prepare_filter(self, filters: Filter):
        # LSM-Vec client does not support filter now.
        self.expr_ = ""

    def _resetDbDirIfNeeded(self) -> None:
        os.makedirs(self.db_dir_, exist_ok=True)

    def _openDb(self, reinit: bool = False) -> None:
        self._resetDbDirIfNeeded()

        import lsm_vec  # pybind module

        opts = lsm_vec.LSMVecDBOptions()
        # _setAttrIfExists(opts, "dim", self.dim_)
        # _setAttrIfExists(opts, "vector_file_path", self.vector_file_path_)
        opts.dim = int(self.dim_)
        opts.reinit = bool(reinit)
        opts.vector_file_path = os.path.abspath(self.vector_file_path_)
        opts.log_file_path = os.path.abspath(os.path.join(self.db_dir_, "lsmvec.log"))


        # _applyBestEffortIndexOptions(opts, self.case_config_)
        # _applyBestEffortStorageOptions(opts, self.db_config_)

        self.db_ = lsm_vec.LSMVecDB.open(self.db_dir_, opts)

    def _closeDb(self) -> None:
        if self.db_ is None:
            return
        try:
            if hasattr(self.db_, "close"):
                self.db_.close()
        except Exception as e:
            log.debug(f"{self.name} close ignored: {e}")
        finally:
            self.db_ = None

    @contextmanager
    def init(self) -> None:
        """
        Open database handle in each process, then close it.
        """
        self._openDb(reinit=False)
        try:
            yield
        finally:
            self._closeDb()

    def insert_embeddings(
        self,
        embeddings: Iterable[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
        **kwargs,
    ) -> tuple[int, Exception]:
        assert self.db_ is not None
        insertCount = 0
        try:
            if self.need_clean_ and not self.cleaned_:
                self._closeDb()
                if os.path.exists(self.db_dir_):
                    shutil.rmtree(self.db_dir_, ignore_errors=True)
                os.makedirs(self.db_dir_, exist_ok=True)
                self._openDb(reinit=True)
                self.cleaned_ = True
            # Best effort batch insert if binding provides it
            if hasattr(self.db_, "insert_batch"):
                self.db_.insert_batch(metadata, list(embeddings))
                insertCount = len(metadata)
                return insertCount, None

            # Fallback: single insert
            embList = [list(map(float, v)) for v in embeddings]
            if len(embList) != len(metadata):
                raise ValueError("len(embeddings) must equal len(metadata)")
            for i in range(len(embList)):
                self.db_.insert(int(metadata[i]), embList[i])
                insertCount += 1
            return insertCount, None
        except Exception as e:
            log.info(f"{self.name} insert failed: {e}")
            return insertCount, e

    def optimize(self, data_size: int | None = None):
        """
        Called between insertion and search.

        LSM-Vec builds graph as inserts happen. We do best effort finalize steps if exposed.
        """
        assert self.db_ is not None

        # Best effort methods, ignore if not present
        for methodName in ["flush", "sync", "compact", "finalize", "build_index", "buildIndex"]:
            if hasattr(self.db_, methodName):
                try:
                    getattr(self.db_, methodName)()
                except Exception as e:
                    log.debug(f"{self.name} optimize method {methodName} ignored: {e}")

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        timeout: int | None = None,
    ) -> list[int]:
        assert self.db_ is not None
        import lsm_vec

        searchOpts = _makeSearchOptions(lsm_vec, int(k), self.case_config_)

        # Best effort query API names
        if hasattr(self.db_, "search_knn"):
            results = self.db_.search_knn(query, searchOpts)
        elif hasattr(self.db_, "searchKnn"):
            results = self.db_.searchKnn(query, searchOpts)
        else:
            raise RuntimeError("LSM-Vec binding does not expose search_knn/searchKnn")

        # results is list of objects with .id
        ids: list[int] = []
        for r in results:
            if hasattr(r, "id"):
                ids.append(int(r.id))
            elif isinstance(r, (tuple, list)) and len(r) > 0:
                ids.append(int(r[0]))
        return ids
