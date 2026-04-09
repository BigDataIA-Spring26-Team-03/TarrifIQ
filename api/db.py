import os
import snowflake.connector
from snowflake.connector import SnowflakeConnection


def get_snowflake_conn() -> SnowflakeConnection:
    """
    Returns a new Snowflake connection using environment variables.
    Caller is responsible for closing the connection.

    Usage:
        conn = get_snowflake_conn()
        try:
            cur = conn.cursor()
            ...
        finally:
            conn.close()
    """
    return snowflake.connector.connect(
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
        database=os.environ["SNOWFLAKE_DATABASE"],
        schema=os.environ["SNOWFLAKE_SCHEMA"],
    )