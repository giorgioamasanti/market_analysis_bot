"""
Central config for phase 2: instrument universe, date ranges, storage paths.
Keeping this in one place means fetch/clean/analysis scripts all agree on
what "the universe" is, and you can add new instruments in one line.
"""

from pathlib import Path

# --- storage ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"       # untouched, as-downloaded parquet
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"   # aligned, forward-filled, analysis-ready

RAW_DIR.mkdir(parents=True, exist_ok=True)
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

# --- date range ---
START_DATE = "2018-01-01"
END_DATE = None  # None = up to today

# --- instrument universe ---
# Grouped so you can test cointegration within a group (where there's an
# economic reason to expect a relationship) rather than firing Johansen
# at a random basket and p-hacking your way to a "result".
UNIVERSE = {
    "energy": ["XLE", "XOP", "USO"],
    "oil_majors": ["XOM", "CVX", "BP", "SHEL"],
    "financials": ["XLF", "KBE", "KRE"],
    "banks": ["JPM", "BAC", "WFC", "C"],
    "gold": ["GLD", "GDX", "GDXJ"],
    "silver_gold": ["SLV", "GLD", "GDX"],
    "semis": ["SMH", "SOXX"],
    "tech_megacap": ["QQQ", "XLK", "VGT"],
    "broad_market": ["SPY", "IVV", "VOO"],
    "airlines": ["DAL", "UAL", "AAL", "LUV"],
    "homebuilders": ["DHI", "LEN", "PHM", "NVR"],
    "retailers": ["WMT", "TGT", "COST"],
    "auto": ["GM", "F", "TM"],
    "healthcare": ["XLV", "IHI", "IBB"],
    "staples": ["XLP", "KO", "PEP"],
    "country_etfs": ["EWJ", "EWG", "EWU", "EWC"],
    "bonds": ["TLT", "IEF", "SHY", "AGG"],
    "reits": ["VNQ", "IYR", "SCHH"],
}

def all_tickers() -> list[str]:
    """Flat, deduplicated list of every ticker in the universe."""
    seen = []
    for group in UNIVERSE.values():
        for t in group:
            if t not in seen:
                seen.append(t)
    return seen