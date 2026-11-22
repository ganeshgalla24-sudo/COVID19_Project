"""
Step 2: Global trend analysis with interactive Plotly charts.

Questions answered
  - How did confirmed cases, deaths, recoveries and active cases grow worldwide?
  - How many new cases per day, and are they speeding up or slowing down?
  - How fast is the infection rate growing, and how long does it take to double?
  - How does the infection rate compare with the recovery rate?
  - Which WHO regions drive the numbers?

Reads the tables made by Step 1. Saves interactive charts (.html) and, if the
'kaleido' package works on your machine, static copies (.png), plus CSV tables.

Run on its own:
    python src/trends.py
or from main.py:
    from src.trends import run_trends
"""

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PROCESSED_DIR = "data/processed"
FIG_DIR = "outputs/figures"
RES_DIR = "outputs/results"

# One colour per measure, used in every chart so the story reads consistently
COLORS = {"confirmed": "#1f77b4", "active": "#ff7f0e",
          "recovered": "#2ca02c", "deaths": "#d62728"}
TEMPLATE = "plotly_white"
TREND_START = "2020-03-01"   # growth rates before March are noisy (tiny numbers)

_png_warned = False


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def save_figure(fig: go.Figure, name: str) -> None:
    """Save an interactive HTML file, plus a PNG copy when kaleido works."""
    global _png_warned
    os.makedirs(FIG_DIR, exist_ok=True)
    html_path = os.path.join(FIG_DIR, f"{name}.html")
    fig.write_html(html_path, include_plotlyjs="cdn")
    msg = f"  saved: {html_path}"
    try:
        fig.write_image(os.path.join(FIG_DIR, f"{name}.png"), width=1100, height=600, scale=2)
        msg += " (+ png)"
    except Exception:
        if not _png_warned:
            print("  note: PNG export skipped (kaleido/Chrome not available). HTML charts are still saved.")
            _png_warned = True
    print(msg)


def _save_table(df: pd.DataFrame, name: str, index: bool = False) -> None:
    os.makedirs(RES_DIR, exist_ok=True)
    df.to_csv(os.path.join(RES_DIR, name), index=index)


def load_tables(processed_dir: str = PROCESSED_DIR):
    world = pd.read_csv(os.path.join(processed_dir, "daily_global.csv"), parse_dates=["date"])
    country = pd.read_csv(os.path.join(processed_dir, "country_daily.csv"), parse_dates=["date"])
    return world, country


def add_growth_features(world: pd.DataFrame) -> pd.DataFrame:
    """Average daily growth over the last 7 days, and the doubling time it implies."""
    w = world.sort_values("date").copy()
    ratio = w["confirmed"] / w["confirmed"].shift(7)
    w["growth_7d_pct"] = (ratio ** (1 / 7) - 1) * 100            # average % growth per day
    w["doubling_days"] = np.log(2) / np.log(1 + w["growth_7d_pct"] / 100)
    return w


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------
def chart_cumulative(w: pd.DataFrame) -> None:
    """Running totals of confirmed, active, recovered and deaths."""
    fig = go.Figure()
    for col in ["confirmed", "active", "recovered", "deaths"]:
        fig.add_trace(go.Scatter(x=w["date"], y=w[col], name=col.capitalize(),
                                 mode="lines", line=dict(color=COLORS[col], width=3)))
    fig.update_layout(title="World: cumulative COVID-19 cases (22 Jan to 27 Jul 2020)",
                      yaxis_title="number of people", hovermode="x unified", template=TEMPLATE)
    fig.update_yaxes(rangemode="tozero")
    save_figure(fig, "01_world_cumulative")


def chart_cases_breakdown(w: pd.DataFrame) -> None:
    """Where do confirmed cases stand: still active, recovered or dead."""
    fig = go.Figure()
    for col in ["deaths", "recovered", "active"]:
        fig.add_trace(go.Scatter(x=w["date"], y=w[col], name=col.capitalize(),
                                 mode="lines", stackgroup="one",
                                 line=dict(width=0.5, color=COLORS[col])))
    fig.update_layout(title="World: what happened to confirmed cases (stacked, total = confirmed)",
                      yaxis_title="number of people", hovermode="x unified", template=TEMPLATE)
    save_figure(fig, "02_world_case_breakdown")


def chart_daily_new(w: pd.DataFrame) -> None:
    """Daily new confirmed, deaths and recovered, with a 7-day average line."""
    items = [("daily_confirmed", "New confirmed per day", "confirmed"),
             ("daily_recovered", "New recovered per day", "recovered"),
             ("daily_deaths", "New deaths per day", "deaths")]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=[t for _, t, _ in items])
    for i, (col, _, key) in enumerate(items, start=1):
        fig.add_trace(go.Bar(x=w["date"], y=w[col], marker_color=COLORS[key], opacity=0.35,
                             name=f"{key} (daily)", showlegend=False), row=i, col=1)
        fig.add_trace(go.Scatter(x=w["date"], y=w[f"{col}_7d_avg"], mode="lines",
                                 line=dict(color=COLORS[key], width=3),
                                 name=f"{key} (7-day average)", showlegend=False), row=i, col=1)
    fig.update_layout(title="World: daily new cases (bars) and 7-day average (line)",
                      height=800, template=TEMPLATE, hovermode="x unified")
    save_figure(fig, "03_world_daily_new")


def chart_growth_and_doubling(w: pd.DataFrame) -> None:
    """How fast is the infection spreading, and how long to double."""
    d = w[w["date"] >= TREND_START]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=["Average daily growth of confirmed cases (last 7 days, %)",
                                        "Doubling time of confirmed cases (days)"])
    fig.add_trace(go.Scatter(x=d["date"], y=d["growth_7d_pct"], mode="lines",
                             line=dict(color=COLORS["confirmed"], width=3), showlegend=False),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=d["date"], y=d["doubling_days"], mode="lines",
                             line=dict(color=COLORS["deaths"], width=3), showlegend=False),
                  row=2, col=1)
    fig.update_layout(title="World: speed of infection (from 1 March 2020)",
                      height=700, template=TEMPLATE, hovermode="x unified")
    fig.update_yaxes(title_text="% per day", row=1, col=1)
    fig.update_yaxes(title_text="days", row=2, col=1)
    save_figure(fig, "04_world_growth_doubling")


def chart_infection_vs_recovery(w: pd.DataFrame) -> None:
    """Rate of infection (new cases) against rate of recovery (new recoveries)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=w["date"], y=w["daily_confirmed_7d_avg"], name="New confirmed (7-day avg)",
                             line=dict(color=COLORS["confirmed"], width=3)))
    fig.add_trace(go.Scatter(x=w["date"], y=w["daily_recovered_7d_avg"], name="New recovered (7-day avg)",
                             line=dict(color=COLORS["recovered"], width=3)))
    fig.add_trace(go.Scatter(x=w["date"], y=w["daily_deaths_7d_avg"], name="New deaths (7-day avg)",
                             line=dict(color=COLORS["deaths"], width=3)))
    fig.update_layout(title="World: infection rate vs recovery rate (per day, 7-day average)",
                      yaxis_title="people per day", hovermode="x unified", template=TEMPLATE)
    save_figure(fig, "05_world_infection_vs_recovery")


def chart_rates(w: pd.DataFrame) -> None:
    """Recovery rate and death rate as a share of confirmed cases."""
    d = w[w["date"] >= TREND_START]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["date"], y=d["recovery_rate"] * 100, name="Recovery rate (%)",
                             line=dict(color=COLORS["recovered"], width=3)))
    fig.add_trace(go.Scatter(x=d["date"], y=d["death_rate"] * 100, name="Death rate (%)",
                             line=dict(color=COLORS["deaths"], width=3)))
    fig.update_layout(title="World: recovery rate and death rate (% of confirmed cases)",
                      yaxis_title="% of confirmed", hovermode="x unified", template=TEMPLATE)
    save_figure(fig, "06_world_recovery_death_rates")


def chart_regions(country: pd.DataFrame) -> pd.DataFrame:
    """Confirmed cases by WHO region over time, and the latest split."""
    reg = (country.groupby(["who_region", "date"], as_index=False)
                  [["confirmed", "deaths", "recovered", "active"]].sum())
    fig = go.Figure()
    for region, g in reg.groupby("who_region"):
        fig.add_trace(go.Scatter(x=g["date"], y=g["confirmed"], name=region, mode="lines",
                                 line=dict(width=3)))
    fig.update_layout(title="Confirmed cases by WHO region", yaxis_title="confirmed cases",
                      hovermode="x unified", template=TEMPLATE)
    save_figure(fig, "07_regions_confirmed")

    last = reg[reg["date"] == reg["date"].max()].set_index("who_region")
    last["share_of_confirmed_%"] = 100 * last["confirmed"] / last["confirmed"].sum()
    last["death_rate_%"] = 100 * last["deaths"] / last["confirmed"]
    last["recovery_rate_%"] = 100 * last["recovered"] / last["confirmed"]
    last = last.drop(columns="date").sort_values("confirmed", ascending=False)
    return last


# --------------------------------------------------------------------------
# Tables and key numbers
# --------------------------------------------------------------------------
def milestones(w: pd.DataFrame) -> pd.DataFrame:
    """Date when the world first passed round numbers of confirmed cases."""
    rows, prev = [], None
    for m in [1_000, 10_000, 100_000, 1_000_000, 5_000_000, 10_000_000, 15_000_000]:
        hit = w[w["confirmed"] >= m]
        if hit.empty:
            continue
        date = hit["date"].iloc[0]
        rows.append({"confirmed_cases": m, "date": date.date(),
                     "days_since_previous": None if prev is None else (date - prev).days})
        prev = date
    return pd.DataFrame(rows)


def weekly_table(w: pd.DataFrame) -> pd.DataFrame:
    """New cases, deaths and recoveries per calendar week, and change vs previous week."""
    wk = (w.set_index("date")[["daily_confirmed", "daily_deaths", "daily_recovered"]]
            .resample("W").sum())
    wk["confirmed_change_vs_prev_week_%"] = 100 * wk["daily_confirmed"].pct_change()
    return wk.reset_index().rename(columns={"date": "week_ending"})


def print_key_findings(w: pd.DataFrame, ms: pd.DataFrame, wk: pd.DataFrame, reg: pd.DataFrame) -> None:
    last = w.iloc[-1]
    peak_c = w.loc[w["daily_confirmed"].idxmax()]
    peak_d = w.loc[w["daily_deaths"].idxmax()]
    peak_c7 = w.loc[w["daily_confirmed_7d_avg"].idxmax()]
    full = wk.iloc[1:-1]   # drop first and last week, which are partial

    print("\n=== Key numbers ===")
    print(f"Last day ({last['date'].date()}): confirmed {int(last['confirmed']):,}, "
          f"deaths {int(last['deaths']):,}, recovered {int(last['recovered']):,}, active {int(last['active']):,}")
    print(f"Highest single day of new cases : {int(peak_c['daily_confirmed']):,} on {peak_c['date'].date()}")
    print(f"Highest 7-day average of new cases: {int(round(peak_c7['daily_confirmed_7d_avg'])):,} on {peak_c7['date'].date()}")
    print(f"Highest single day of new deaths: {int(peak_d['daily_deaths']):,} on {peak_d['date'].date()}")
    print(f"Average daily growth, last 7 days: {last['growth_7d_pct']:.2f}% -> doubling time {last['doubling_days']:.0f} days")
    d = w[w['date'] == '2020-04-01'].iloc[0]
    print(f"Doubling time on 1 April: {d['doubling_days']:.1f} days (growth {d['growth_7d_pct']:.1f}% per day)")
    print(f"Recovery rate {last['recovery_rate']:.1%}, death rate {last['death_rate']:.1%}")
    best = full.loc[full["daily_confirmed"].idxmax()]
    print(f"Busiest full week: ending {best['week_ending'].date()} with {int(best['daily_confirmed']):,} new cases")
    print("\nMilestones (confirmed cases):")
    print(ms.to_string(index=False))
    print("\nWHO regions on the last day:")
    print(reg[["confirmed", "deaths", "share_of_confirmed_%", "death_rate_%", "recovery_rate_%"]]
          .round(2).to_string())


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------
def run_trends(world: pd.DataFrame = None, country: pd.DataFrame = None,
               processed_dir: str = PROCESSED_DIR) -> None:
    if world is None or country is None:
        world, country = load_tables(processed_dir)
    w = add_growth_features(world)

    print("\nBuilding charts...")
    chart_cumulative(w)
    chart_cases_breakdown(w)
    chart_daily_new(w)
    chart_growth_and_doubling(w)
    chart_infection_vs_recovery(w)
    chart_rates(w)
    reg = chart_regions(country)

    ms, wk = milestones(w), weekly_table(w)
    _save_table(ms, "world_milestones.csv")
    _save_table(wk, "world_weekly.csv")
    _save_table(reg, "region_summary.csv", index=True)
    _save_table(w, "world_trend_table.csv")
    print_key_findings(w, ms, wk, reg)
    print("\nTrends finished. Charts in outputs/figures, tables in outputs/results.")


if __name__ == "__main__":
    run_trends()