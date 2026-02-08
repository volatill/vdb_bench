from __future__ import annotations

from typing import Annotated, Unpack, TypedDict

import click

from vectordb_bench.backend.clients import DB
from vectordb_bench.cli.cli import (
    CommonTypedDict,
    cli,
    click_parameter_decorators_from_typed_dict,
    get_custom_case_config,
    run,
)


class LsmVecTypedDict(CommonTypedDict):
    db_root: Annotated[
        str,
        click.option("--db-root", type=str, default="./run/lsmvec", show_default=True),
    ]
    vector_file_name: Annotated[
        str,
        click.option("--vector-file-name", type=str, default="vector.log", show_default=True),
    ]
    always_clean_start: Annotated[
        bool,
        click.option("--always-clean-start/--no-always-clean-start", default=True, show_default=True),
    ]
    vec_storage: Annotated[
        int,
        click.option("--vec-storage", type=int, default=0, show_default=True),
    ]
    paged_cache_pages: Annotated[
        int,
        click.option("--paged-cache-pages", type=int, default=256, show_default=True),
    ]


# 注意：这里必须是 TypedDict 风格，并且每个字段类型必须是 Annotated
class LsmVecIndexTypedDict(TypedDict):
    M: Annotated[int, click.option("--M", "M", type=int, default=8, show_default=True)]
    Mmax: Annotated[int, click.option("--Mmax", "Mmax", type=int, default=16, show_default=True)]
    Ml: Annotated[int, click.option("--Ml", "Ml", type=int, default=1, show_default=True)]
    efc: Annotated[int, click.option("--efc", "efc", type=int, default=64, show_default=True)]
    ef_search: Annotated[int, click.option("--ef-search", "ef_search", type=int, default=0, show_default=True)]

@cli.command(name="lsmvec")
@click_parameter_decorators_from_typed_dict(LsmVecTypedDict)
@click_parameter_decorators_from_typed_dict(LsmVecIndexTypedDict)
def lsmvec(**parameters: Unpack[LsmVecTypedDict]):
    """
    Benchmark runner for LSM-Vec.
    """
    from .config import LSMVecConfig, LSMVecIndexConfig

    parameters["custom_case"] = get_custom_case_config(parameters)

    efSearch = parameters.get("ef_search", 0)
    caseConfig = LSMVecIndexConfig(
        M=parameters["M"],
        Mmax=parameters["Mmax"],
        Ml=parameters["Ml"],
        efc=parameters["efc"],
        ef_search=None if int(efSearch) == 0 else int(efSearch),
    )

    dbConfig = LSMVecConfig(
        db_label=parameters["db_label"],
        db_root=parameters["db_root"],
        vector_file_name=parameters["vector_file_name"],
        always_clean_start=parameters["always_clean_start"],
        vec_storage=parameters["vec_storage"],
        paged_cache_pages=parameters["paged_cache_pages"],
    )

    run(
        db=DB.LsmVec,  # 如果你的 Enum 里名字不是 LSMVec，把这里改成实际名字
        db_config=dbConfig,
        db_case_config=caseConfig,
        **parameters,
    )
