from src.data_prep import run_data_prep
from src.trends import run_trends
from src.country_analysis import run_country_analysis
from src.forecast import run_forecast

if __name__ == "__main__":
    tables = run_data_prep("data/raw/covid_19_clean_complete.csv")
    run_trends(tables["daily_global"], tables["country_daily"])
    run_country_analysis(tables["country_daily"], tables["country_summary"],
                         tables["daily_india"], tables["daily_global"])
    run_forecast({"World": tables["daily_global"], "India": tables["daily_india"]})