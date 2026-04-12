"""
Run all Snowflake migrations in order from snowflake/migrations/*.sql
"""

import logging
import os
from pathlib import Path

from snowflake.connection import get_connection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)


def _parse_sql_statements(sql: str):
    """Parse SQL statements from a string, handling comments and semicolons."""
    # Remove comments
    lines = []
    for line in sql.split('\n'):
        # Remove inline comments
        if '--' in line:
            line = line[:line.index('--')]
        lines.append(line)

    sql_clean = '\n'.join(lines)

    # Split on semicolon, but only actual statement terminators
    statements = []
    current = []
    for char in sql_clean:
        if char == ';':
            stmt = ''.join(current).strip()
            if stmt:
                statements.append(stmt)
            current = []
        else:
            current.append(char)

    # Add any remaining statement
    stmt = ''.join(current).strip()
    if stmt:
        statements.append(stmt)

    return statements


def run_migrations():
    """Execute all .sql files in snowflake/migrations/ in alphabetical order."""
    migrations_dir = Path(__file__).parent / "migrations"

    if not migrations_dir.exists():
        logger.error(f"Migrations directory not found: {migrations_dir}")
        return False

    migration_files = sorted(migrations_dir.glob("*.sql"))

    if not migration_files:
        logger.warning("No migration files found")
        return True

    conn = get_connection()
    cur = conn.cursor()

    try:
        for migration_file in migration_files:
            logger.info(f"Running migration: {migration_file.name}")

            with open(migration_file, 'r') as f:
                sql = f.read()

            statements = _parse_sql_statements(sql)
            logger.debug(f"Parsed {len(statements)} statements")

            for statement in statements:
                logger.debug(f"Executing: {statement[:60]}...")
                cur.execute(statement)

            logger.info(f"✓ {migration_file.name} completed")

        conn.commit()
        logger.info("All migrations completed successfully")
        return True

    except Exception as exc:
        logger.error(f"Migration failed: {exc}", exc_info=True)
        conn.rollback()
        return False

    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    success = run_migrations()
    exit(0 if success else 1)
