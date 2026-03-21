from typing import Annotated, Unpack

import click
from pydantic import SecretStr

from vectordb_bench.backend.clients import DB, EmptyDBCaseConfig
from vectordb_bench.cli.cli import (
    CommonTypedDict,
    cli,
    click_parameter_decorators_from_typed_dict,
    run,
)


class PineconeTypedDict(CommonTypedDict):
    api_key: Annotated[
        str,
        click.option("--api-key", type=str, help="Pinecone API key", required=True),
    ]
    index_name: Annotated[
        str,
        click.option("--index-name", type=str, help="Pinecone index name", required=True),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(PineconeTypedDict)
def Pinecone(**parameters: Unpack[PineconeTypedDict]):
    from .config import PineconeConfig

    run(
        db=DB.Pinecone,
        db_config=PineconeConfig(
            db_label=parameters["db_label"],
            api_key=SecretStr(parameters["api_key"]),
            index_name=parameters["index_name"],
        ),
        db_case_config=EmptyDBCaseConfig(),
        **parameters,
    )
