"""
Step 3: Country comparison and India deep dive (interactive Plotly charts).

Questions answered
  - Which countries are hit hardest, and where are cases rising or falling fastest?
  - Which countries have high death rates or high recovery rates?
  - How did India's outbreak develop, and how does it compare with the world and
    with the other big countries?

Reads the tables made by Step 1. Saves interactive charts (.html), PNG copies if
'kaleido' works, and CSV tables.

Run on its own:
    python src/country_analysis.py
or from main.py:
    from src.country_analysis import run_country_analysis

Note: the small helper functions (save_figure, growth features) are repeated from
trends.py on purpose, so that each step file can run on its own.
"""

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PROCESSED_DIR = "data/processed"
FIG_DIR = "outputs/figures"
RES_DIR = "outputs/results"

COLORS = {"confirmed": "#1f77b4", "active": "#ff7f0e",
          "recovered": "#2ca02c", "deaths": "#d62728"}
TEMPLATE = "plotly_white"
PEERS = ["US", "Brazil", "India", "Russia", "South Africa"]
MIN_FOR_RATES = 10_000      # countries with fewer confirmed cases are left out of rate comparisons
MIN_PREV_WEEK = 1_000       # ... and out of "fastest growing" if they had fewer new cases the week before
RANK_FROM_CASES = 100       # India's rank is shown from the day it passed 100 confirmed cases
NO_RECOVERY_REPORTING = 0.01  # recovery rate below 1% usually means recoveries are not reported

# Names in the dataset that Plotly's map does not recognise
MAP_NAMES = {"US": "United States", "Korea, South": "South Korea", "Taiwan*": "Taiwan",
             "Burma": "Myanmar", "Congo (Brazzaville)": "Republic of the Congo",
             "Congo (Kinshasa)": "Democratic Republic of the Congo",
             "Czechia": "Czech Republic", "Cabo Verde": "Cape Verde",
             "Eswatini": "Swaziland", "North Macedonia": "Macedonia",
             "Cote d'Ivoire": "Ivory Coast", "Holy See": "Vatican"}

_png_warned = False


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def save_figure(fig: go.Figure, name: str) -> None:
    global _png_warned
    os.makedirs(FIG_DIR, exist_ok=True)
    html_path = os.path.join(FIG_DIR, f"{name}.html")
    fig.write_html(html_path, include_plotlyjs="cdn")
    msg = f"  saved: {html_path}"
    try:
        fig.write_image(os.path.join(FIG_DIR, f"{name}.png"), width=1100, height=600, scale=2)
        msg += " (+ png)"
    except Exception:
        msg += " (png skipped)"
        if not _png_warned:
            print("  note: some PNG exports were skipped (kaleido/Chrome or internet not available, "
                  "maps often need internet). The HTML charts are still saved.")
            _png_warned = True
    print(msg)


def _save_table(df: pd.DataFrame, name: str, index: bool = False) -> None:
    os.makedirs(RES_DIR, exist_ok=True)
    df.to_csv(os.path.join(RES_DIR, name), index=index)


def load_tables(processed_dir: str = PROCESSED_DIR):
    country = pd.read_csv(os.path.join(processed_dir, "country_daily.csv"), parse_dates=["date"])
    summary = pd.read_csv(os.path.join(processed_dir, "country_summary.csv"), index_col="country")
    india = pd.read_csv(os.path.join(processed_dir, "daily_india.csv"), parse_dates=["date"])
    world = pd.read_csv(os.path.join(processed_dir, "daily_global.csv"), parse_dates=["date"])
    return country, summary, india, world


def add_growth_features(df: pd.DataFrame) -> pd.DataFrame:
    """Average daily growth over the last 7 days and the doubling time it implies."""
    d = df.sort_values("date").copy()
    ratio = d["confirmed"] / d["confirmed"].shift(7)
    d["growth_7d_pct"] = (ratio ** (1 / 7) - 1) * 100
    d["doubling_days"] = np.log(2) / np.log(1 + d["growth_7d_pct"] / 100)
    return d


def milestones(df: pd.DataFrame, levels) -> pd.DataFrame:
    rows, prev = [], None
    for m in levels:
        hit = df[df["confirmed"] >= m]
        if hit.empty:
            continue
        date = hit["date"].iloc[0]
        rows.append({"confirmed_cases": m, "date": date.date(),
                     "days_since_previous": None if prev is None else (date - prev).days})
        prev = date
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Part A: country comparison
# --------------------------------------------------------------------------
def chart_top10(summary: pd.DataFrame) -> pd.DataFrame:
    top = summary.head(10).copy()
    fig = go.Figure()
    for col in ["deaths", "recovered", "active"]:
        fig.add_trace(go.Bar(y=top.index, x=top[col], name=col.capitalize(), orientation="h",
                             marker_color=COLORS[col]))
    fig.update_layout(barmode="stack", title="Top 10 countries by confirmed cases (27 Jul 2020)",
                      xaxis_title="confirmed cases (split by outcome)", template=TEMPLATE,
                      yaxis=dict(autorange="reversed"), hovermode="y unified")
    save_figure(fig, "08_top10_countries")
    return top


def chart_world_map(summary: pd.DataFrame) -> None:
    s = summary.copy()
    names = [MAP_NAMES.get(c, c) for c in s.index]
    fig = go.Figure(go.Choropleth(
        locations=names, locationmode="country names",
        z=np.log10(s["confirmed"].clip(lower=1)),
        text=[f"{c}<br>confirmed: {int(v):,}<br>deaths: {int(d):,}"
              for c, v, d in zip(s.index, s["confirmed"], s["deaths"])],
        hoverinfo="text", colorscale="Reds",
        colorbar=dict(title="confirmed", tickvals=[1, 2, 3, 4, 5, 6],
                      ticktext=["10", "100", "1K", "10K", "100K", "1M"])))
    fig.update_layout(title="Confirmed cases by country (colour on a log scale)",
                      template=TEMPLATE, geo=dict(showframe=False, projection_type="natural earth"))
    save_figure(fig, "09_world_map_confirmed")


def chart_top10_trends(country: pd.DataFrame, top10: list) -> None:
    fig = go.Figure()
    for c in top10:
        d = country[country["country"] == c]
        fig.add_trace(go.Scatter(x=d["date"], y=d["daily_confirmed_7d_avg"], name=c,
                                 mode="lines", line=dict(width=3 if c == "India" else 2)))
    fig.update_layout(title="New confirmed cases per day (7-day average): top 10 countries",
                      yaxis_title="new cases per day", hovermode="x unified", template=TEMPLATE)
    save_figure(fig, "10_top10_daily_trends")


def fastest_changing(summary: pd.DataFrame) -> pd.DataFrame:
    eligible = summary[summary["new_cases_prev_7d"] >= MIN_PREV_WEEK].copy()
    eligible = eligible.sort_values("weekly_change_pct", ascending=False)
    print(f"[growth] {len(eligible)} countries had at least {MIN_PREV_WEEK:,} new cases in the week before")
    up, down = eligible.head(10), eligible.tail(10).iloc[::-1]

    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.2,
                        subplot_titles=["Cases rising fastest", "Cases falling fastest"])
    fig.add_trace(go.Bar(y=up.index, x=up["weekly_change_pct"], orientation="h",
                         marker_color=COLORS["deaths"], showlegend=False), row=1, col=1)
    fig.add_trace(go.Bar(y=down.index, x=down["weekly_change_pct"], orientation="h",
                         marker_color=COLORS["recovered"], showlegend=False), row=1, col=2)
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title_text="% change in weekly new cases")
    fig.update_layout(title="Change in new cases: last 7 days vs the 7 days before",
                      template=TEMPLATE, height=550)
    save_figure(fig, "11_fastest_changing_countries")
    return eligible


def chart_death_vs_recovery(summary: pd.DataFrame) -> pd.DataFrame:
    s = summary[summary["confirmed"] >= MIN_FOR_RATES].copy()
    fig = go.Figure(go.Scatter(
        x=s["death_rate"] * 100, y=s["recovery_rate"] * 100, mode="markers",
        marker=dict(size=np.sqrt(s["confirmed"]) / 25 + 5, color=COLORS["confirmed"], opacity=0.5,
                    line=dict(width=1, color="white")),
        text=[f"{c}<br>confirmed: {int(v):,}" for c, v in zip(s.index, s["confirmed"])],
        hovertemplate="%{text}<br>death rate: %{x:.1f}%<br>recovery rate: %{y:.1f}%<extra></extra>"))
    fig.update_layout(title=f"Death rate vs recovery rate (countries with {MIN_FOR_RATES:,}+ confirmed; bubble = size of outbreak)",
                      xaxis_title="death rate (% of confirmed)", yaxis_title="recovery rate (% of confirmed)",
                      template=TEMPLATE)
    save_figure(fig, "12_death_vs_recovery_rate")
    return s


# --------------------------------------------------------------------------
# Part B: India deep dive
# --------------------------------------------------------------------------
def chart_india_cumulative(i: pd.DataFrame) -> None:
    fig = go.Figure()
    for col in ["confirmed", "active", "recovered", "deaths"]:
        fig.add_trace(go.Scatter(x=i["date"], y=i[col], name=col.capitalize(), mode="lines",
                                 line=dict(color=COLORS[col], width=3)))
    fig.update_layout(title="India: cumulative COVID-19 cases", yaxis_title="number of people",
                      hovermode="x unified", template=TEMPLATE)
    save_figure(fig, "13_india_cumulative")


def chart_india_daily(i: pd.DataFrame) -> None:
    items = [("daily_confirmed", "New confirmed per day", "confirmed"),
             ("daily_recovered", "New recovered per day", "recovered"),
             ("daily_deaths", "New deaths per day", "deaths")]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=[t for _, t, _ in items])
    for r, (col, _, key) in enumerate(items, start=1):
        fig.add_trace(go.Bar(x=i["date"], y=i[col], marker_color=COLORS[key], opacity=0.35,
                             showlegend=False), row=r, col=1)
        fig.add_trace(go.Scatter(x=i["date"], y=i[f"{col}_7d_avg"], mode="lines",
                                 line=dict(color=COLORS[key], width=3), showlegend=False), row=r, col=1)
    fig.update_layout(title="India: daily new cases (bars) and 7-day average (line)",
                      height=800, template=TEMPLATE, hovermode="x unified")
    save_figure(fig, "14_india_daily_new")


def chart_india_growth(i: pd.DataFrame, start: str = "2020-03-15") -> None:
    d = i[i["date"] >= start]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=["Average daily growth of confirmed cases (last 7 days, %)",
                                        "Doubling time of confirmed cases (days)"])
    fig.add_trace(go.Scatter(x=d["date"], y=d["growth_7d_pct"], mode="lines",
                             line=dict(color=COLORS["confirmed"], width=3), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=d["date"], y=d["doubling_days"], mode="lines",
                             line=dict(color=COLORS["deaths"], width=3), showlegend=False), row=2, col=1)
    fig.update_layout(title="India: speed of infection (from 15 March 2020)",
                      height=700, template=TEMPLATE, hovermode="x unified")
    fig.update_yaxes(title_text="% per day", row=1, col=1)
    fig.update_yaxes(title_text="days", row=2, col=1)
    save_figure(fig, "15_india_growth_doubling")


def chart_india_vs_peers(country: pd.DataFrame) -> None:
    fig = go.Figure()
    for c in PEERS:
        d = country[country["country"] == c]
        fig.add_trace(go.Scatter(x=d["date"], y=d["daily_confirmed_7d_avg"], name=c, mode="lines",
                                 line=dict(width=4 if c == "India" else 2)))
    fig.update_layout(title="India vs the other largest outbreaks: new cases per day (7-day average)",
                      yaxis_title="new cases per day", hovermode="x unified", template=TEMPLATE)
    save_figure(fig, "16_india_vs_peers")


def india_rank_and_share(country: pd.DataFrame, i: pd.DataFrame, world: pd.DataFrame):
    """India's rank by confirmed cases and its share of the world total, over time."""
    pivot = country.pivot(index="date", columns="country", values="confirmed")
    rank = pivot.rank(axis=1, ascending=False, method="min")["India"]
    share = 100 * pivot["India"] / world.set_index("date")["confirmed"]
    out = pd.DataFrame({"india_rank": rank, "india_share_of_world_%": share}).reset_index()
    # Ranks are only meaningful once India has a real outbreak. On day 1 most countries
    # have zero cases and tie, which would give India a misleadingly high rank.
    out = out.merge(i[["date", "confirmed"]], on="date")
    out = out[out["confirmed"] >= RANK_FROM_CASES].reset_index(drop=True)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
                        subplot_titles=["India's rank by confirmed cases (1 = most cases)",
                                        "India's share of world confirmed cases (%)"])
    d = out
    fig.add_trace(go.Scatter(x=d["date"], y=d["india_rank"], mode="lines",
                             line=dict(color=COLORS["confirmed"], width=3, shape="hv"),
                             showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=d["date"], y=d["india_share_of_world_%"], mode="lines",
                             line=dict(color=COLORS["active"], width=3), showlegend=False), row=2, col=1)
    fig.update_yaxes(autorange="reversed", row=1, col=1)
    fig.update_layout(title="India's position in the world", height=700, template=TEMPLATE,
                      hovermode="x unified")
    save_figure(fig, "17_india_rank_and_share")
    return out


# --------------------------------------------------------------------------
# Key numbers
# --------------------------------------------------------------------------
def print_findings(summary, top10, eligible, rates, i, ms, rank_df) -> None:
    print("\n=== Top 10 countries (27 Jul 2020) ===")
    print(top10[["confirmed", "deaths", "recovered", "active", "death_rate", "recovery_rate",
                 "share_of_global_confirmed"]].round(4).to_string())
    print(f"Top 3 countries hold {top10['share_of_global_confirmed'].head(3).sum():.1%} of all cases; "
          f"top 10 hold {top10['share_of_global_confirmed'].sum():.1%}")

    print("\n=== Weekly change in new cases (countries with enough cases) ===")
    print("Rising fastest:")
    print(eligible.head(5)[["new_cases_prev_7d", "new_cases_last_7d", "weekly_change_pct"]].round(1).to_string())
    print("Falling fastest:")
    print(eligible.tail(5).iloc[::-1][["new_cases_prev_7d", "new_cases_last_7d", "weekly_change_pct"]].round(1).to_string())
    print(f"Countries with rising cases: {int((eligible['weekly_change_pct'] > 0).sum())} of {len(eligible)}")

    print(f"\n=== Death and recovery rates (countries with {MIN_FOR_RATES:,}+ confirmed: {len(rates)}) ===")
    print("Highest death rate:")
    print(rates.sort_values("death_rate", ascending=False).head(5)[["confirmed", "death_rate", "recovery_rate"]].round(3).to_string())
    silent = rates[rates["recovery_rate"] < NO_RECOVERY_REPORTING]
    print(f"Recovery rate below {NO_RECOVERY_REPORTING:.0%} (most likely NOT reporting recoveries, "
          f"so their 'active' cases are overstated): {len(silent)} countries")
    print("  " + ", ".join(silent.sort_values("confirmed", ascending=False).index))
    reporting = rates[rates["recovery_rate"] >= NO_RECOVERY_REPORTING]
    print("Lowest recovery rate among countries that do report recoveries:")
    print(reporting.sort_values("recovery_rate").head(5)[["confirmed", "death_rate", "recovery_rate"]].round(3).to_string())

    last = i.iloc[-1]
    peak = i.loc[i["daily_confirmed"].idxmax()]
    print("\n=== India ===")
    print(f"Last day: confirmed {int(last['confirmed']):,}, deaths {int(last['deaths']):,}, "
          f"recovered {int(last['recovered']):,}, active {int(last['active']):,}")
    print(f"Recovery rate {last['recovery_rate']:.1%}, death rate {last['death_rate']:.1%}")
    print(f"Highest single day of new cases: {int(peak['daily_confirmed']):,} on {peak['date'].date()}")
    print(f"7-day average of new cases: {last['daily_confirmed_7d_avg']:,.0f} per day (last day)")
    print(f"Average daily growth, last 7 days: {last['growth_7d_pct']:.2f}% -> doubling time {last['doubling_days']:.0f} days")
    i_apr = i[i["date"] == "2020-04-01"].iloc[0]
    i_jun = i[i["date"] == "2020-06-01"].iloc[0]
    print(f"Doubling time: {i_apr['doubling_days']:.1f} days on 1 April, {i_jun['doubling_days']:.1f} days on 1 June")
    print("Milestones (confirmed cases):")
    print(ms.to_string(index=False))
    for r in [10, 5, 3]:
        hit = rank_df[rank_df["india_rank"] <= r]
        if not hit.empty:
            print(f"India first in the top {r}: {hit['date'].iloc[0].date()}")
    print(f"India's rank on the last day: {int(rank_df['india_rank'].iloc[-1])}, "
          f"share of world cases: {rank_df['india_share_of_world_%'].iloc[-1]:.1f}%")
    w = i.set_index("date")["daily_confirmed"].resample("W").sum()
    print(f"India's last full week: {int(w.iloc[-2]):,} new cases ({100 * (w.iloc[-2] / w.iloc[-3] - 1):+.0f}% vs the week before)")


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------
def run_country_analysis(country=None, summary=None, india=None, world=None,
                         processed_dir: str = PROCESSED_DIR) -> None:
    if any(x is None for x in (country, summary, india, world)):
        country, summary, india, world = load_tables(processed_dir)
    i = add_growth_features(india)

    print("\nBuilding country charts...")
    top10 = chart_top10(summary)
    chart_world_map(summary)
    chart_top10_trends(country, list(top10.index))
    eligible = fastest_changing(summary)
    rates = chart_death_vs_recovery(summary)

    print("\nBuilding India charts...")
    chart_india_cumulative(i)
    chart_india_daily(i)
    chart_india_growth(i)
    chart_india_vs_peers(country)
    rank_df = india_rank_and_share(country, i, world)

    ms = milestones(i, [100, 1_000, 10_000, 100_000, 500_000, 1_000_000])
    _save_table(top10, "top10_countries.csv", index=True)
    _save_table(eligible, "weekly_change_by_country.csv", index=True)
    _save_table(rates, "country_rates.csv", index=True)
    _save_table(i, "india_trend_table.csv")
    _save_table(ms, "india_milestones.csv")
    _save_table(rank_df, "india_rank_and_share.csv")
    print_findings(summary, top10, eligible, rates, i, ms, rank_df)
    print("\nCountry analysis finished. Charts in outputs/figures, tables in outputs/results.")


if __name__ == "__main__":
    run_country_analysis()