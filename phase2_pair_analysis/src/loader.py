"""
Thin query layer over the clean parquet file.

Right now this is overkill for one small parquet file - you could just
pd.read_parquet() it directly, and get_prices() below does exactly that.
The DuckDB path is here because it's the same interface you'll want once
data/clean/ has many partitioned files (e.g. per-instrument order book
data in phase 4): DuckDB queries parquet directly on disk without
loading it all into memory first, which pandas can't do.
"""

import duckdb
import pandas as pd

from src.config import CLEAN_DIR

CLEAN_TABLE = CLEAN_DIR / "adj_close_wide.parquet"


def get_prices(tickers: list[str] | None = None,
                start: str | None = None,
                end: str | None = None) -> pd.DataFrame:
    """Query the clean adjusted-close table with optional column/date filters."""
    con = duckdb.connect()

    cols = "*" if tickers is None else ", ".join(['"Date"'] + [f'"{t}"' for t in tickers])
    query = f'SELECT {cols} FROM read_parquet(\'{CLEAN_TABLE}\')'

    conditions = []
    if start:
        conditions.append(f'"Date" >= \'{start}\'')
    if end:
        conditions.append(f'"Date" <= \'{end}\'')
    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    df = con.execute(query).df()
    return df.set_index("Date")


if __name__ == "__main__":
    # quick smoke test
    df = get_prices(tickers=["SPY", "IVV", "VOO"])
    print(df.tail())