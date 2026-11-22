"""
Step 4: Forecast the next 7 days with Facebook Prophet (world and India).

What this script does
  1. For confirmed cases, deaths and recovered patients (world and India) it
     forecasts the number of NEW cases per day for the next 7 days with Prophet,
     then adds them to the last known running total. That gives the expected
     running total a week from the last date in the data (27 Jul 2020 -> 3 Aug 2020).
  2. Measures how accurate this approach is with a BACKTEST: it pretends to be
     standing at 6 earlier weekly dates, forecasts the following 7 days, and
     compares with what really happened. Two simple baselines are tested too.
  3. Draws interactive Plotly charts that combine the real data and the forecast.

Why forecast daily new cases and not the running total directly?
  Prophet fits a trend plus seasonality. Running totals are a steep accelerating
  curve and Prophet's straight-line trend lags behind it (we tested this: its
  7-day error was about twice as large). Daily new cases are much better suited,
  and they show a clear weekly pattern (fewer reports at weekends) which Prophet
  captures with its weekly seasonality.

Run on its own:
    python src/forecast.py
or from main.py:
    from src.forecast import run_forecast
"""

import logging
import os
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prophet import Prophet

# Prophet/cmdstanpy print a lot of progress lines; keep the console readable
logging.getLogger("cmdstanpy").disabled = True
logging.getLogger("prophet").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

PROCESSED_DIR = "data/processed"
FIG_DIR = "outputs/figures"
RES_DIR = "outputs/results"

HORIZON = 7              # days to forecast
TRAIN_WINDOW = 45        # Prophet learns from the last 45 days only (recent trend matters most)
CHANGEPOINT_PRIOR = 0.5  # how freely the trend may bend (higher = follows recent changes more)
INTERVAL_WIDTH = 0.95    # width of the uncertainty band
N_SAMPLES = 500          # simulated futures used to build the band
N_ORIGINS = 6            # number of weekly backtest dates
SEED = 42

TARGETS = ["confirmed", "deaths", "recovered"]
DAILY_COL = {"confirmed": "daily_confirmed", "deaths": "daily_deaths", "recovered": "daily_recovered"}
COLORS = {"confirmed": "#1f77b4", "deaths": "#d62728", "recovered": "#2ca02c"}
TEMPLATE = "plotly_white"

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
        fig.write_image(os.path.join(FIG_DIR, f"{name}.png"), width=1100, height=fig.layout.height or 600, scale=2)
        msg += " (+ png)"
    except Exception:
        msg += " (png skipped)"
        if not _png_warned:
            print("  note: some PNG exports were skipped (kaleido/Chrome not available). "
                  "The HTML charts are still saved.")
            _png_warned = True
    print(msg)


def _save_table(df: pd.DataFrame, name: str, index: bool = False) -> None:
    os.makedirs(RES_DIR, exist_ok=True)
    df.to_csv(os.path.join(RES_DIR, name), index=index)


def load_tables(processed_dir: str = PROCESSED_DIR):
    world = pd.read_csv(os.path.join(processed_dir, "daily_global.csv"), parse_dates=["date"])
    india = pd.read_csv(os.path.join(processed_dir, "daily_india.csv"), parse_dates=["date"])
    return {"World": world, "India": india}


def get_series(df: pd.DataFrame, target: str):
    """Return (running total, daily new) as Prophet-style tables with columns ds and y."""
    cum = df[["date", target]].rename(columns={"date": "ds", target: "y"})
    day = df[["date", DAILY_COL[target]]].rename(columns={"date": "ds", DAILY_COL[target]: "y"})
    return cum, day


# --------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------
def prophet_forecast(cum: pd.DataFrame, day: pd.DataFrame, origin: pd.Timestamp,
                     horizon: int = HORIZON) -> pd.DataFrame:
    """
    Stand at `origin` (the last day of known data) and forecast the next `horizon` days.
    Returns one row per future day: expected running total, band (lower/upper) and
    expected new cases that day.
    """
    train = day[(day["ds"] <= origin) & (day["ds"] > origin - pd.Timedelta(days=TRAIN_WINDOW))]
    model = Prophet(growth="linear", yearly_seasonality=False, daily_seasonality=False,
                    weekly_seasonality=True, changepoint_prior_scale=CHANGEPOINT_PRIOR,
                    interval_width=INTERVAL_WIDTH, uncertainty_samples=N_SAMPLES)
    model.fit(train)

    future = model.make_future_dataframe(periods=horizon)
    np.random.seed(SEED)                                   # same band every run
    sims = model.predictive_samples(future)["yhat"][-horizon:, :]   # (horizon, N_SAMPLES)
    sims = np.clip(sims, 0, None)                           # new cases cannot be negative

    start = cum.loc[cum["ds"] == origin, "y"].iloc[0]       # last REPORTED running total
    total = start + np.cumsum(sims, axis=0)                 # running total in every simulated future
    lo, hi = (1 - INTERVAL_WIDTH) / 2 * 100, (1 + INTERVAL_WIDTH) / 2 * 100
    return pd.DataFrame({
        "ds": future["ds"].iloc[-horizon:].values,
        "yhat": total.mean(axis=1),
        "yhat_lower": np.percentile(total, lo, axis=1),
        "yhat_upper": np.percentile(total, hi, axis=1),
        "new_per_day": sims.mean(axis=1),
    })


def baseline_forecasts(cum: pd.DataFrame, origin: pd.Timestamp, horizon: int = HORIZON) -> dict:
    """Two simple methods that Prophet has to beat."""
    hist = cum[cum["ds"] <= origin]["y"].values
    avg_increase = (hist[-1] - hist[-8]) / 7
    return {"persistence (repeat last total)": np.repeat(hist[-1], horizon),
            "linear (last 7-day average)": hist[-1] + avg_increase * np.arange(1, horizon + 1)}


# --------------------------------------------------------------------------
# Backtest
# --------------------------------------------------------------------------
def backtest(data: dict):
    """Forecast 7 days from several past dates and compare with what really happened."""
    last = data["World"]["date"].max()
    origins = [last - pd.Timedelta(days=HORIZON * k) for k in range(N_ORIGINS, 0, -1)]
    rows, last_week = [], {}

    for entity, df in data.items():
        for target in TARGETS:
            cum, day = get_series(df, target)
            for o in origins:
                actual = cum[(cum["ds"] > o) & (cum["ds"] <= o + pd.Timedelta(days=HORIZON))]["y"].values
                fc = prophet_forecast(cum, day, o)
                methods = {"Prophet": fc["yhat"].values, **baseline_forecasts(cum, o)}
                for name, pred in methods.items():
                    ape = np.abs(pred - actual) / actual * 100
                    rows.append({
                        "entity": entity, "target": target, "origin": o.date(), "method": name,
                        "MAPE_%": ape.mean(), "day7_error_%": ape[-1],
                        "MAE_people": np.abs(pred - actual).mean(),
                        "inside_band_%": (100 * ((actual >= fc["yhat_lower"].values) &
                                                 (actual <= fc["yhat_upper"].values)).mean()
                                          if name == "Prophet" else np.nan)})
                if o == origins[-1]:
                    last_week[(entity, target)] = (fc, actual)
    return pd.DataFrame(rows), last_week


def summarise_backtest(bt: pd.DataFrame):
    by_series = (bt.groupby(["entity", "target", "method"])[["MAPE_%", "day7_error_%", "inside_band_%"]]
                   .mean().reset_index())
    overall = (bt.groupby("method")[["MAPE_%", "day7_error_%", "MAE_people", "inside_band_%"]]
                 .mean().sort_values("MAPE_%"))
    print(f"\n=== Backtest: {N_ORIGINS} weekly start dates, 7-day forecasts, average over all 6 series ===")
    print("(MAPE = average % error over the 7 days; day7 = % error on the 7th day; "
          "inside_band = % of days the real value fell inside the Prophet band)")
    print(overall.round(2).to_string())

    print("\nMAPE (%) by series:")
    print(by_series.pivot_table(index=["entity", "target"], columns="method", values="MAPE_%")
          .round(2).to_string())
    wins = (by_series.pivot_table(index=["entity", "target"], columns="method", values="MAPE_%")
            .pipe(lambda t: (t["Prophet"] < t["linear (last 7-day average)"]).sum()))
    print(f"Prophet beats the linear baseline in {wins} of 6 series.")
    return by_series, overall


# --------------------------------------------------------------------------
# Final forecast
# --------------------------------------------------------------------------
def final_forecast(data: dict):
    forecasts, rows = {}, []
    for entity, df in data.items():
        origin = df["date"].max()
        for target in TARGETS:
            cum, day = get_series(df, target)
            fc = prophet_forecast(cum, day, origin)
            forecasts[(entity, target)] = (cum, fc, origin)
            f = fc.copy()
            f.insert(0, "target", target)
            f.insert(0, "entity", entity)
            rows.append(f)
    table = pd.concat(rows, ignore_index=True)
    return forecasts, table


def print_forecast_summary(forecasts: dict) -> pd.DataFrame:
    rows = []
    for (entity, target), (cum, fc, origin) in forecasts.items():
        last_total = cum.loc[cum["ds"] == origin, "y"].iloc[0]
        last7_new = (last_total - cum.loc[cum["ds"] == origin - pd.Timedelta(days=7), "y"].iloc[0]) / 7
        end = fc.iloc[-1]
        rows.append({"entity": entity, "target": target,
                     "last_total": last_total, "forecast_date": pd.Timestamp(end["ds"]).date(),
                     "forecast_total": end["yhat"], "band_low": end["yhat_lower"], "band_high": end["yhat_upper"],
                     "added_in_7_days": end["yhat"] - last_total,
                     "new_per_day_forecast": fc["new_per_day"].mean(), "new_per_day_last_7d": last7_new})
    s = pd.DataFrame(rows)
    s["change_in_daily_new_%"] = 100 * (s["new_per_day_forecast"] / s["new_per_day_last_7d"] - 1)
    print("\n=== Forecast: expected running totals 7 days after the last day of data ===")
    show = s.copy()
    for c in ["last_total", "forecast_total", "band_low", "band_high", "added_in_7_days",
              "new_per_day_forecast", "new_per_day_last_7d"]:
        show[c] = show[c].round(0).astype(int).map("{:,}".format)
    show["change_in_daily_new_%"] = show["change_in_daily_new_%"].round(1)
    print(show.to_string(index=False))
    return s


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------
def _add_forecast_traces(fig, cum, fc, origin, color, row, col, show_legend):
    """Real data line, forecast line and uncertainty band for one panel."""
    last_val = cum.loc[cum["ds"] == origin, "y"].iloc[0]
    x_fc = [origin] + list(fc["ds"])
    band_up = [last_val] + list(fc["yhat_upper"])
    band_lo = [last_val] + list(fc["yhat_lower"])
    mid = [last_val] + list(fc["yhat"])

    fig.add_trace(go.Scatter(x=cum["ds"], y=cum["y"], mode="lines", name="Actual",
                             line=dict(color=color, width=3), legendgroup="actual",
                             showlegend=show_legend), row=row, col=col)
    fig.add_trace(go.Scatter(x=x_fc, y=band_up, mode="lines", line=dict(width=0),
                             hoverinfo="skip", showlegend=False), row=row, col=col)
    fig.add_trace(go.Scatter(x=x_fc, y=band_lo, mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(120,120,120,0.25)", name=f"{INTERVAL_WIDTH:.0%} band",
                             legendgroup="band", showlegend=show_legend), row=row, col=col)
    fig.add_trace(go.Scatter(x=x_fc, y=mid, mode="lines+markers", name="Forecast",
                             line=dict(color="black", width=3, dash="dash"), marker=dict(size=6),
                             legendgroup="fc", showlegend=show_legend), row=row, col=col)


def chart_forecast(entity: str, forecasts: dict) -> None:
    fig = make_subplots(rows=3, cols=2, vertical_spacing=0.09, horizontal_spacing=0.1,
                        subplot_titles=[f"{t.capitalize()}: full history" if c == 1
                                        else f"{t.capitalize()}: last 30 days + 7-day forecast"
                                        for t in TARGETS for c in (1, 2)])
    for r, target in enumerate(TARGETS, start=1):
        cum, fc, origin = forecasts[(entity, target)]
        _add_forecast_traces(fig, cum, fc, origin, COLORS[target], r, 1, show_legend=(r == 1))
        recent = cum[cum["ds"] > origin - pd.Timedelta(days=30)]
        _add_forecast_traces(fig, recent, fc, origin, COLORS[target], r, 2, show_legend=False)
    fig.update_layout(title=f"{entity}: COVID-19 running totals and 7-day Prophet forecast",
                      height=950, template=TEMPLATE, hovermode="x unified",
                      legend=dict(orientation="h", y=1.07))
    name = "18_forecast_world" if entity == "World" else "19_forecast_india"
    save_figure(fig, name)


def chart_backtest_error(by_series: pd.DataFrame) -> None:
    t = by_series.pivot_table(index=["entity", "target"], columns="method", values="MAPE_%")
    labels = [f"{e} {tg}" for e, tg in t.index]
    fig = go.Figure()
    for method, color in [("Prophet", "#1f77b4"), ("linear (last 7-day average)", "#ff7f0e"),
                          ("persistence (repeat last total)", "#bbbbbb")]:
        fig.add_trace(go.Bar(x=labels, y=t[method], name=method, marker_color=color))
    fig.update_layout(barmode="group", title="Backtest: average 7-day forecast error (lower is better)",
                      yaxis_title="average error (% of actual)", template=TEMPLATE, height=600)
    save_figure(fig, "20_backtest_error")


def chart_backtest_last_week(last_week: dict, data: dict) -> None:
    keys = [(e, t) for e in ("World", "India") for t in TARGETS]
    fig = make_subplots(rows=2, cols=3, vertical_spacing=0.14, horizontal_spacing=0.08,
                        subplot_titles=[f"{e} {t}" for e, t in keys])
    for i, (entity, target) in enumerate(keys):
        r, c = i // 3 + 1, i % 3 + 1
        fc, actual = last_week[(entity, target)]
        dates = pd.to_datetime(fc["ds"])
        fig.add_trace(go.Scatter(x=dates, y=actual, mode="lines+markers", name="Actual",
                                 line=dict(color=COLORS[target], width=3), showlegend=(i == 0),
                                 legendgroup="a"), row=r, col=c)
        fig.add_trace(go.Scatter(x=dates, y=fc["yhat_upper"], mode="lines", line=dict(width=0),
                                 hoverinfo="skip", showlegend=False), row=r, col=c)
        fig.add_trace(go.Scatter(x=dates, y=fc["yhat_lower"], mode="lines", line=dict(width=0),
                                 fill="tonexty", fillcolor="rgba(120,120,120,0.25)",
                                 name=f"{INTERVAL_WIDTH:.0%} band", showlegend=(i == 0), legendgroup="b"),
                      row=r, col=c)
        fig.add_trace(go.Scatter(x=dates, y=fc["yhat"], mode="lines+markers", name="Forecast",
                                 line=dict(color="black", width=2, dash="dash"), showlegend=(i == 0),
                                 legendgroup="f"), row=r, col=c)
    fig.update_layout(title="Backtest: the forecast made 7 days before the end of the data vs what happened",
                      height=700, template=TEMPLATE, legend=dict(orientation="h", y=1.1))
    save_figure(fig, "21_backtest_last_week")


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------
def run_forecast(data: dict = None, processed_dir: str = PROCESSED_DIR) -> None:
    if data is None:
        data = load_tables(processed_dir)

    print("Running backtest (this takes about a minute)...")
    bt, last_week = backtest(data)
    by_series, overall = summarise_backtest(bt)
    _save_table(bt, "forecast_backtest_details.csv")
    _save_table(by_series, "forecast_backtest_by_series.csv")
    _save_table(overall, "forecast_backtest_overall.csv", index=True)

    print("\nFitting final models...")
    forecasts, table = final_forecast(data)
    _save_table(table, "forecast_next_7_days.csv")
    summary = print_forecast_summary(forecasts)
    _save_table(summary, "forecast_summary.csv")

    print("\nBuilding charts...")
    chart_forecast("World", forecasts)
    chart_forecast("India", forecasts)
    chart_backtest_error(by_series)
    chart_backtest_last_week(last_week, data)
    print("\nForecast finished. Charts in outputs/figures, tables in outputs/results.")


if __name__ == "__main__":
    run_forecast()