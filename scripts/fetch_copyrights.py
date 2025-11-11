#!/usr/bin/env python3
"""
Fetch copyright statements from a PostgreSQL (Fossology) database and write to CSV.

Usage:
  python scripts/fetch_copyrights.py --limit 1000 --start-date 2025-01-01 --end-date 2025-10-31 --output out.csv --batch-size 1000

The script uses fetchmany internally with the specified batch-size to control memory usage
when fetching large result sets. Results are accumulated in memory and written to CSV at the end.

Environment variables (required):
  DB_NAME, DB_HOST, DB_PORT, DB_PASSWORD
Optional env:
  DB_USER (defaults to 'postgres')
"""

import argparse
import os
import logging
import sys
from datetime import datetime

try:
    import pandas as pd
    import psycopg2
except Exception as e:
    print("Missing dependency. Please install dev dependencies (psycopg2-binary).", file=sys.stderr)
    raise

logger = logging.getLogger("fetch_copyrights")


SQL_QUERY_TEMPLATE = """
-- User will insert their specific SQL query here.
-- It should select relevant copyright data and support filtering by date and limiting results.
-- Example (adapt to your schema):
-- SELECT copyright_text AS text, date, label FROM copyrights WHERE date BETWEEN %s AND %s LIMIT %s;
"""


def get_db_conn():
    dbname = os.environ.get("DB_NAME")
    host = os.environ.get("DB_HOST")
    port = os.environ.get("DB_PORT")
    password = os.environ.get("DB_PASSWORD")
    user = os.environ.get("DB_USER", "postgres")

    missing = [k for k, v in [("DB_NAME", dbname), ("DB_HOST", host), ("DB_PORT", port), ("DB_PASSWORD", password)] if not v]
    if missing:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}")

    conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port)
    return conn


def run_query(conn, query, params=None):
    # Run a query and fetch results in batches using fetchmany to avoid memory pressure.
    with conn.cursor() as cur:
        cur.execute(query, params or ())
        cols = [desc[0] for desc in cur.description]
        rows = []
        while True:
            batch = cur.fetchmany(cur.arraysize or 1000)
            if not batch:
                break
            rows.extend(batch)
    return cols, rows


def main():
    parser = argparse.ArgumentParser(description="Fetch copyrights from PostgreSQL and save to CSV")
    parser.add_argument("--limit", type=int, default=1000, help="Max number of rows to fetch")
    parser.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", default=None, help="Output CSV filename (optional)")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for fetchmany (controls memory usage)")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    try:
        sd = datetime.strptime(args.start_date, "%Y-%m-%d")
        ed = datetime.strptime(args.end_date, "%Y-%m-%d")
    except ValueError:
        logger.error("start-date and end-date must be in YYYY-MM-DD format")
        sys.exit(2)

    if args.output:
        out_fname = args.output
    else:
        out_fname = f"copyright_data_{args.start_date}_to_{args.end_date}.csv"

    try:
        conn = get_db_conn()
    except Exception as e:
        logger.exception("Failed to connect to DB: %s", e)
        sys.exit(1)

    try:
        # Set the arraysize for fetchmany batching
        with conn.cursor() as cur:
            cur.arraysize = args.batch_size

        params = (args.start_date, args.end_date, args.limit)
        cols, rows = run_query(conn, SQL_QUERY_TEMPLATE, params=params)
        df = pd.DataFrame(rows, columns=cols)
        df.to_csv(out_fname, index=False)
        logger.info("Wrote %d rows to %s", len(df), out_fname)
    except Exception as e:
        logger.exception("Query execution failed: %s", e)
        sys.exit(1)
    finally:
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
