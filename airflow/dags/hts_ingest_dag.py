"""
Weekly USITC HTS ingestion into Snowflake HTS_CODES (idempotent, 500-row batches).
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator


def _ingest_hts_codes() -> None:
    from ingestion.hts_idempotent_load import run_hts_ingest_airflow

    run_hts_ingest_airflow()


DEFAULT_ARGS = {
    "owner": "tariffiq",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}


with DAG(
    dag_id="hts_ingest",
    default_args=DEFAULT_ARGS,
    description="USITC HTS (two API passes) → Snowflake HTS_CODES, idempotent upsert",
    schedule="@weekly",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["tariffiq", "hts", "snowflake"],
) as dag:
    PythonOperator(
        task_id="ingest_hts_codes",
        python_callable=_ingest_hts_codes,
    )
