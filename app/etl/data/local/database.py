import re
from abc import ABC, abstractmethod
from enum import Enum
from typing import final, override

import sqlalchemy
import pandas as pd

from app.etl.data.base_data_types import (
    FieldPathBase,
    IExtractor,
    ILoader,
)


class DatabaseTypes(Enum):
    MSSQL = "mssql"
    SQLITE = "sqlite"


# A single unquoted SQL identifier: letters/digits/underscore, not starting
# with a digit. Table names are matched against this (optionally as
# `schema.table`) before ever being interpolated into SQL text, so that a
# query like `{sqlite:db.sqlite|users UNION SELECT * FROM secrets --}`
# is rejected outright instead of being executed.
_IDENTIFIER_PART = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class IDatabase(FieldPathBase, IExtractor, ILoader, ABC):
    def __init__(self, path: str):
        FieldPathBase.__init__(self, path)
        self.table_name: str = None  # type: ignore
        self.engine: sqlalchemy.Engine = None  # type: ignore
        self.initialize_connection()
        self._validate_table_name()

    @abstractmethod
    def initialize_connection(self) -> None:
        pass

    def _validate_table_name(self) -> None:
        parts = (self.table_name or "").split(".")
        if not parts or not all(_IDENTIFIER_PART.match(p) for p in parts):
            raise ValueError(
                f"Invalid table name {self.table_name!r} in data source path. "
                "Table names may only contain letters, digits, and "
                "underscores (optionally as schema.table), and cannot "
                "start with a digit or contain spaces, quotes, or "
                "SQL keywords/punctuation."
            )

    def _quoted_table_name(self) -> str:
        preparer = self.engine.dialect.identifier_preparer
        return ".".join(preparer.quote(part) for part in self.table_name.split("."))

    @final
    def extract(self) -> pd.DataFrame:
        return pd.read_sql(
            sqlalchemy.text(f"select * from {self._quoted_table_name()};"),
            self.engine,
        )

    @final
    def load(self, data: pd.DataFrame):
        # pandas' to_sql builds its INSERT through SQLAlchemy's own Table/
        # Column objects, which quote the identifier safely on every
        # supported backend, so no extra quoting is needed here.
        data.to_sql(self.table_name, self.engine, if_exists="append", index=False)


class MSSQLDatabase(IDatabase):
    def __init__(self, path: str):
        IDatabase.__init__(self, path)

    @override
    def initialize_connection(self) -> None:
        connection_string = self.path.split("|")
        server_name = connection_string[0]
        data_base_name = connection_string[1]
        self.table_name = connection_string[2]
        self.engine = sqlalchemy.create_engine(
            f"mssql+pyodbc://@{server_name}/{data_base_name}?trusted_connection=yes&driver=ODBC+Driver+18+for+SQL+Server&Encrypt=no"
        )


class SQLITEDatabase(IDatabase):
    def __init__(self, path: str):
        IDatabase.__init__(self, path)

    @override
    def initialize_connection(self) -> None:
        path_parts = self.path.split("|")
        data_base_name = path_parts[0]
        self.table_name = path_parts[1]
        self.engine = sqlalchemy.create_engine(f"sqlite:///{data_base_name}")
