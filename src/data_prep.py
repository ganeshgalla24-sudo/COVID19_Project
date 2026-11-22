"""
Step 1: Data preparation (load, clean, aggregate, feature engineering).

The raw file has CUMULATIVE totals per location per day (a running total).
This script
  1. loads and cleans the raw data (negative values set to zero),
  2. sums provinces up to country level,
  3. creates DAILY new cases/deaths/recoveries from the running totals,
  4. adds rates, growth and 7-day averages,
  5. saves four tables to data/processed/.

Run on its own:
    python src/data_prep.py
or from main.py:
    from src.data_prep import run_data_prep
"""

import os

import numpy as np
import pandas as pd

RAW_PATH = "data/raw/covid_19_clean_complete.csv"
PROCESSED_DIR = "data/processed"

COUNT_COLS = ["confirmed", "deaths", "recovered", "active"]
DAILY_COLS = ["daily_confirmed", "daily_deaths", "daily_recovered"]


# --------------------------------------------------------------------------
# 1. Load
# --------------------------------------------------------------------------
def load_data(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[load] raw shape: {df.shape}")
    return df


# --------------------------------------------------------------------------
# 2. Clean
# --------------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns, parse dates, remove duplicates, set negative counts to zero."""
    df = df.copy()

    df = df.rename(columns={
        "Province/State": "province_state", "Country/Region": "country",
        "Lat": "lat", "Long": "long", "Date": "date", "Confirmed": "confirmed",
        "Deaths": "deaths", "Recovered": "recovered", "Active": "active",
        "WHO Region": "who_region",
    })
    df["date"] = pd.to_datetime(df["date"])
    for col in ["province_state", "country", "who_region"]:
        df[col] = df[col].astype("string").str.strip()

    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"[clean] removed {n_before - len(df)} duplicate rows")

    # Province/State is empty for countries that report one number: that is normal
    print(f"[clean] province_state is empty in {int(df['province_state'].isna().sum())} rows "
          f"(normal: those countries are not split into provinces)")

    # Negative counts are data errors -> set to zero
    neg = {c: int((df[c] < 0).sum()) for c in COUNT_COLS}
    print(f"[clean] negative values set to zero: {neg}")
    for c in COUNT_COLS:
        df[c] = df[c].clip(lower=0)

    # Report (but keep) rows where recovered > confirmed
    print(f"[clean] rows where recovered > confirmed (kept, reported only): "
          f"{int((df['recovered'] > df['confirmed']).sum())}")

    return df.sort_values(["country", "province_state", "date"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# 3. Aggregate: provinces -> countries
# --------------------------------------------------------------------------
def build_country_level(df: pd.DataFrame) -> pd.DataFrame:
    """One row per country per day (provinces are summed)."""
    out = (df.groupby(["country", "date"], as_index=False)
             .agg(who_region=("who_region", "first"),
                  confirmed=("confirmed", "sum"),
                  deaths=("deaths", "sum"),
                  recovered=("recovered", "sum"),
                  active=("active", "sum")))
    print(f"[aggregate] country-level table: {out.shape} "
          f"({out['country'].nunique()} countries x {out['date'].nunique()} days)")
    return out.sort_values(["country", "date"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# 4. Daily numbers + features
# --------------------------------------------------------------------------
def _safe_divide(num: pd.Series, den: pd.Series) -> pd.Series:
    """Divide, returning 0 where the denominator is 0."""
    return (num / den.replace(0, np.nan)).fillna(0)


def add_daily_and_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add daily new numbers, rates, growth rate and 7-day averages (per country)."""
    df = df.sort_values(["country", "date"]).copy()
    g = df.groupby("country")

    # Daily new = today's running total minus yesterday's.
    # Day 1 has no 'yesterday', so it is set to 0. A dip in a running total
    # (data correction) would give a negative number -> set to zero.
    for total, daily in zip(["confirmed", "deaths", "recovered"], DAILY_COLS):
        df[daily] = g[total].diff().fillna(0).clip(lower=0)

    # Rates (as a share of confirmed cases)
    df["recovery_rate"] = _safe_divide(df["recovered"], df["confirmed"])
    df["death_rate"] = _safe_divide(df["deaths"], df["confirmed"])

    # Daily growth rate of confirmed cases, in % of yesterday's running total
    prev_confirmed = df.groupby("country")["confirmed"].shift(1)
    df["growth_rate_pct"] = 100 * _safe_divide(df["daily_confirmed"], prev_confirmed.fillna(0))

    # 7-day moving averages smooth out reporting noise
    for daily in DAILY_COLS:
        df[f"{daily}_7d_avg"] = (df.groupby("country")[daily]
                                   .transform(lambda s: s.rolling(7, min_periods=1).mean()))
    return df


def build_global(country_df: pd.DataFrame) -> pd.DataFrame:
    """World totals per day. Daily numbers are the sum of the country-level daily numbers."""
    g = (country_df.groupby("date", as_index=False)
                   [["confirmed", "deaths", "recovered", "active"] + DAILY_COLS].sum())
    g["country"] = "World"
    g["who_region"] = "All"
    g = add_daily_and_features_world(g)
    return g


def add_daily_and_features_world(g: pd.DataFrame) -> pd.DataFrame:
    """Rates, growth and 7-day averages for a table that already has daily columns."""
    g = g.sort_values("date").copy()
    g["recovery_rate"] = _safe_divide(g["recovered"], g["confirmed"])
    g["death_rate"] = _safe_divide(g["deaths"], g["confirmed"])
    prev = g["confirmed"].shift(1).fillna(0)
    g["growth_rate_pct"] = 100 * _safe_divide(g["daily_confirmed"], prev)
    for daily in DAILY_COLS:
        g[f"{daily}_7d_avg"] = g[daily].rolling(7, min_periods=1).mean()
    return g.reset_index(drop=True)


# --------------------------------------------------------------------------
# 5. Country summary (latest day)
# --------------------------------------------------------------------------
def build_country_summary(country_df: pd.DataFrame) -> pd.DataFrame:
    last = country_df["date"].max()
    latest = country_df[country_df["date"] == last].set_index("country")
    summary = latest[["who_region", "confirmed", "deaths", "recovered", "active",
                      "recovery_rate", "death_rate"]].copy()

    # Cases in the last 7 days vs the 7 days before that
    d = country_df.set_index("date")
    last_week = d[d.index > last - pd.Timedelta(days=7)].groupby("country")["daily_confirmed"].sum()
    prev_week = d[(d.index <= last - pd.Timedelta(days=7)) &
                  (d.index > last - pd.Timedelta(days=14))].groupby("country")["daily_confirmed"].sum()
    summary["new_cases_last_7d"] = last_week
    summary["new_cases_prev_7d"] = prev_week
    summary["weekly_change_pct"] = 100 * _safe_divide(
        summary["new_cases_last_7d"] - summary["new_cases_prev_7d"], summary["new_cases_prev_7d"])
    summary["share_of_global_confirmed"] = summary["confirmed"] / summary["confirmed"].sum()
    return summary.sort_values("confirmed", ascending=False)


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------
def run_data_prep(raw_path: str = RAW_PATH, out_dir: str = PROCESSED_DIR) -> dict:
    """Full Step 1 pipeline. Returns a dict of the four tables."""
    raw = load_data(raw_path)
    clean = clean_data(raw)
    country = add_daily_and_features(build_country_level(clean))
    world = build_global(country)
    india = country[country["country"] == "India"].reset_index(drop=True)
    summary = build_country_summary(country)

    # Sanity checks
    last = world.iloc[-1]
    print("\n[check] last day:", last["date"].date())
    print(f"[check] world confirmed (reported running total): {int(last['confirmed']):,}")
    print(f"[check] world confirmed (sum of daily numbers)   : {int(world['daily_confirmed'].sum()):,}")
    print(f"[check] India confirmed: {int(india.iloc[-1]['confirmed']):,}, "
          f"deaths: {int(india.iloc[-1]['deaths']):,}, recovered: {int(india.iloc[-1]['recovered']):,}")
    print(f"[check] negative daily values left: "
          f"{int((country[DAILY_COLS] < 0).sum().sum())}")

    os.makedirs(out_dir, exist_ok=True)
    tables = {"country_daily": country, "daily_global": world,
              "daily_india": india, "country_summary": summary}
    for name, t in tables.items():
        t.to_csv(os.path.join(out_dir, f"{name}.csv"), index=(name == "country_summary"))
        print(f"[save] {out_dir}/{name}.csv  shape={t.shape}")
    return tables


if __name__ == "__main__":
    run_data_prep()