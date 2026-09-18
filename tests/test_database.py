from unittest.mock import (
    MagicMock,
    patch,
)

import pytest

from corecoder.backend.database import Database


@pytest.fixture
def database():
    return Database(
        host="127.0.0.1",
        port=3306,
        user="root",
        password="password",
        database="corecoder",
    )


@pytest.fixture
def mysql_connection():
    connection = MagicMock()
    cursor = MagicMock()

    connection.cursor.return_value.__enter__.return_value = (
        cursor
    )

    return connection, cursor


def test_execute_commits_transaction(
    database,
    mysql_connection,
):
    connection, cursor = mysql_connection

    cursor.execute.return_value = 1

    with patch(
        "corecoder.backend.database.pymysql.connect",
        return_value=connection,
    ):
        affected = database.execute(
            """
            INSERT INTO prompt_templates(name)
            VALUES (%s)
            """,
            ("agent_system",),
        )

    assert affected == 1
    cursor.execute.assert_called_once()
    connection.commit.assert_called_once()
    connection.rollback.assert_not_called()
    connection.close.assert_called_once()


def test_fetch_one_returns_dictionary(
    database,
    mysql_connection,
):
    connection, cursor = mysql_connection

    cursor.fetchone.return_value = {
        "id": 1,
        "name": "agent_system",
    }

    with patch(
        "corecoder.backend.database.pymysql.connect",
        return_value=connection,
    ):
        row = database.fetch_one(
            """
            SELECT id, name
            FROM prompt_templates
            WHERE id = %s
            """,
            (1,),
        )

    assert row == {
        "id": 1,
        "name": "agent_system",
    }

    connection.commit.assert_called_once()
    connection.close.assert_called_once()


def test_transaction_rolls_back_on_error(
    database,
    mysql_connection,
):
    connection, cursor = mysql_connection

    cursor.execute.side_effect = RuntimeError(
        "database error"
    )

    with (
        patch(
            "corecoder.backend.database.pymysql.connect",
            return_value=connection,
        ),
        pytest.raises(
            RuntimeError,
            match="database error",
        ),
    ):
        database.execute(
            """
            UPDATE prompt_templates
            SET name = %s
            """,
            ("new-name",),
        )

    connection.commit.assert_not_called()
    connection.rollback.assert_called_once()
    connection.close.assert_called_once()