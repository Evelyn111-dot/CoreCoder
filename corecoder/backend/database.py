from __future__ import annotations

from collections.abc import (
    Generator,
    Sequence,
)
from contextlib import contextmanager
from typing import Any

import pymysql
from pymysql.connections import Connection
from pymysql.cursors import DictCursor


class Database:
    """使用 PyMySQL 直接访问 MySQL。"""

    def __init__(
        self,
        host: str,
        port: int,
        user: str,
        password: str,
        database: str,
    ):
        self._config = {
            "host": host,
            "port": port,
            "user": user,
            "password": password,
            "database": database,
            "charset": "utf8mb4",
            "cursorclass": DictCursor,
            "autocommit": False,
            "connect_timeout": 5,
        }

    def connect(self) -> Connection:
        """创建一个新的 MySQL 连接。"""

        return pymysql.connect(
            **self._config
        )

    @contextmanager
    def transaction(
        self,
    ) -> Generator[Connection, None, None]:
        """成功提交，异常回滚，最后关闭连接。"""

        connection = self.connect()

        try:
            yield connection
            connection.commit()

        except Exception:
            connection.rollback()
            raise

        finally:
            connection.close()

    def execute(
        self,
        sql: str,
        params: Sequence[Any] | None = None,
    ) -> int:
        """执行增删改语句，返回受影响行数。"""

        with (
            self.transaction() as connection,
            connection.cursor() as cursor,
        ):
            return cursor.execute(
                sql,
                params,
            )

    def insert(
        self,
        sql: str,
        params: Sequence[Any] | None = None,
    ) -> int:
        """执行 INSERT 并返回自增主键。"""

        with (
            self.transaction() as connection,
            connection.cursor() as cursor,
        ):
            cursor.execute(
                sql,
                params,
            )

            return int(
                cursor.lastrowid
            )

    def fetch_one(
        self,
        sql: str,
        params: Sequence[Any] | None = None,
    ) -> dict[str, Any] | None:
        """查询一条数据。"""

        with (
            self.transaction() as connection,
            connection.cursor() as cursor,
        ):
            cursor.execute(
                sql,
                params,
            )

            return cursor.fetchone()

    def fetch_all(
        self,
        sql: str,
        params: Sequence[Any] | None = None,
    ) -> list[dict[str, Any]]:
        """查询多条数据。"""

        with (
            self.transaction() as connection,
            connection.cursor() as cursor,
        ):
            cursor.execute(
                sql,
                params,
            )

            return list(
                cursor.fetchall()
            )

    def ping(self) -> bool:
        """检查 MySQL 是否可以连接。"""

        row = self.fetch_one(
            "SELECT 1 AS healthy"
        )

        return bool(
            row
            and row["healthy"] == 1
        )