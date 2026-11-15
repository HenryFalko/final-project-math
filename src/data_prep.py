"""Builds the country-year panel from the two Kaggle datasets.

Sources
-------
* EM-DAT (via Kaggle, brsdincer/all-natural-disasters-19002021-eosdis):
  one row = one natural disaster (country, year, type, deaths, people affected, damages).
* World Bank (via Kaggle, zgrcemta/world-gdpgdp-gdp-per-capita-and-annual-growths):
  current GDP (USD), GDP per capita, annual real GDP growth (%).

Unit of observation: (country, year) from 1971 to 2020.
Target: annual real GDP growth (%) in year t.

Every transformation that "learns" something from the data (winsorization
bounds, standardization means / standard deviations) is estimated on the
training sample only -> no data leakage.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")

# Climate-related disasters (EM-DAT classification: hydrological,
# meteorological, climatological). Earthquakes / volcanoes (geophysical) are
# used as a comparison group: they are not driven by the climate.
CLIMATE_GROUPS = {
    "Flood": "hydro",
    "Landslide": "hydro",
    "Glacial lake outburst": "hydro",
    "Storm": "meteo",
    "Extreme temperature": "climato",
    "Drought": "climato",
    "Wildfire": "climato",
}
GEO_TYPES = {"Earthquake", "Volcanic activity", "Mass movement (dry)"}

DMG_COL = "Total Damages ('000 US$)"

# Final explanatory variables (in this order).
DISASTER_FEATURES = [
    "log_dmg_climate_pct",       # climate damages as % of GDP (log1p), year t
    "log_dmg_climate_pct_lag1",  # same, year t-1 (delayed effect)
    "log_deaths_climate_pm",     # deaths per million inhabitants (log1p)
    "log_affected_climate_pct",  # people affected as % of population (log1p)
    "n_hydro",                   # number of floods / landslides
    "n_meteo",                   # number of storms
    "n_climato",                 # number of droughts / heatwaves / wildfires
    "log_dmg_geo_pct",           # geophysical damages (control / comparison)
]
CONTROL_FEATURES = [
    "growth_lag1",     # previous year's growth (persistence)
    "log_gdppc_lag1",  # level of development (log GDP per capita, t-1)
    "world_growth",    # global business cycle in year t
]
FEATURES = DISASTER_FEATURES + CONTROL_FEATURES
TARGET = "gdp_growth"

# Chronological split (past and future are never mixed).
TRAIN_YEARS = (1971, 2005)
VAL_YEARS = (2006, 2012)
TEST_YEARS = (2013, 2019)
STRESS_YEARS = (2020, 2020)  # COVID year, analysed separately


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_disasters(data_dir: str = DATA_DIR) -> pd.DataFrame:
    path = glob.glob(os.path.join(data_dir, "disasters", "DISASTERS", "1970-2021*"))[0]
    return pd.read_csv(path)


def _wb_long(name: str, data_dir: str = DATA_DIR) -> pd.DataFrame:
    """World Bank file (wide format) -> long format (ISO, year, value)."""
    df = pd.read_csv(os.path.join(data_dir, "gdp", f"{name}.csv"))
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    long = df.melt(id_vars=["Country Name", "Code"], var_name="year", value_name=name)
    long["year"] = long["year"].astype(int)
    return long.rename(columns={"Code": "ISO", "Country Name": "country"})


def load_world_bank(data_dir: str = DATA_DIR) -> pd.DataFrame:
    wb = _wb_long("gdp", data_dir)
    for n in ["gdp_growth", "gdp_per_capita"]:
        wb = wb.merge(_wb_long(n, data_dir).drop(columns="country"), on=["ISO", "year"])
    return wb


# ---------------------------------------------------------------------------
# Aggregating disasters by country-year
# ---------------------------------------------------------------------------
def aggregate_disasters(dis: pd.DataFrame) -> pd.DataFrame:
    d = dis.copy()
    d["group"] = d["Disaster Type"].map(CLIMATE_GROUPS)
    d.loc[d["Disaster Type"].isin(GEO_TYPES), "group"] = "geo"
    d = d[d["group"].notna()]
    d["is_climate"] = d["group"] != "geo"
    # Missing value = not reported in EM-DAT; counted as 0
    # (a strong assumption discussed in the report: reporting bias).
    for c in ["Total Deaths", "Total Affected", DMG_COL]:
        d[c] = d[c].fillna(0.0)

    key = ["ISO", "Year"]
    clim = d[d.is_climate].groupby(key).agg(
        dmg_climate=(DMG_COL, "sum"),
        deaths_climate=("Total Deaths", "sum"),
        affected_climate=("Total Affected", "sum"),
    )
    counts = d.pivot_table(index=key, columns="group", values="Dis No",
                           aggfunc="count", fill_value=0)
    counts.columns = [f"n_{c}" for c in counts.columns]
    geo = d[d.group == "geo"].groupby(key).agg(dmg_geo=(DMG_COL, "sum"))
    agg = clim.join(counts, how="outer").join(geo, how="outer").fillna(0.0)
    agg = agg.reset_index().rename(columns={"Year": "year"})
    for g in ["hydro", "meteo", "climato", "geo"]:
        if f"n_{g}" not in agg:
            agg[f"n_{g}"] = 0.0
    return agg


# ---------------------------------------------------------------------------
# Full panel
# ---------------------------------------------------------------------------
def build_panel(data_dir: str = DATA_DIR) -> pd.DataFrame:
    dis = load_disasters(data_dir)
    wb = load_world_bank(data_dir)

    world = wb.loc[wb.ISO == "WLD", ["year", "gdp_growth"]].rename(
        columns={"gdp_growth": "world_growth"})

    # Country universe = countries present in EM-DAT (excludes World Bank
    # aggregates such as "Euro area", "World"...).
    countries = set(dis["ISO"].unique())
    wb = wb[wb.ISO.isin(countries)].copy()

    agg = aggregate_disasters(dis)
    panel = wb.merge(agg, on=["ISO", "year"], how="left")
    fill_cols = [c for c in agg.columns if c not in ("ISO", "year")]
    panel[fill_cols] = panel[fill_cols].fillna(0.0)  # no recorded disaster
    panel = panel.merge(world, on="year", how="left")

    panel["population"] = panel["gdp"] / panel["gdp_per_capita"]
    # Damages (thousands of current USD) relative to GDP (current USD) -> %.
    panel["dmg_climate_pct"] = 100 * panel["dmg_climate"] * 1e3 / panel["gdp"]
    panel["dmg_geo_pct"] = 100 * panel["dmg_geo"] * 1e3 / panel["gdp"]
    panel["deaths_climate_pm"] = 1e6 * panel["deaths_climate"] / panel["population"]
    panel["affected_climate_pct"] = 100 * panel["affected_climate"] / panel["population"]

    for c in ["dmg_climate_pct", "dmg_geo_pct", "deaths_climate_pm", "affected_climate_pct"]:
        panel[f"log_{c}"] = np.log1p(panel[c])

    panel = panel.sort_values(["ISO", "year"])
    g = panel.groupby("ISO")
    # Lags: check that the previous row really is year t-1.
    prev_year = g["year"].shift(1)
    ok = prev_year == panel["year"] - 1
    panel["growth_lag1"] = g["gdp_growth"].shift(1).where(ok)
    panel["log_gdppc_lag1"] = np.log(g["gdp_per_capita"].shift(1)).where(ok)
    panel["log_dmg_climate_pct_lag1"] = g["log_dmg_climate_pct"].shift(1).where(ok)

    panel = panel[(panel.year >= 1971) & (panel.year <= 2020)]
    panel = panel.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)
    return panel


# ---------------------------------------------------------------------------
# Splitting + preprocessing fitted on the training set
# ---------------------------------------------------------------------------
def year_mask(df: pd.DataFrame, years: tuple[int, int]) -> pd.Series:
    return (df.year >= years[0]) & (df.year <= years[1])


@dataclass
class Splits:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    X_stress: np.ndarray
    y_stress: np.ndarray
    features: list
    mean: np.ndarray
    std: np.ndarray
    clip: dict
    frames: dict  # raw DataFrames per split (for analysis)


def make_splits(panel: pd.DataFrame, features=FEATURES, standardize=True,
                winsor=(0.01, 0.99)) -> Splits:
    """Chronological split + winsorization + standardization (parameters = train)."""
    frames = {
        "train": panel[year_mask(panel, TRAIN_YEARS)].copy(),
        "val": panel[year_mask(panel, VAL_YEARS)].copy(),
        "test": panel[year_mask(panel, TEST_YEARS)].copy(),
        "stress": panel[year_mask(panel, STRESS_YEARS)].copy(),
    }
    tr = frames["train"]
    # Winsorization bounds (target + lagged growth) estimated on the training set:
    # a few extreme values (wars, e.g. Iraq, Libya: +/-60 %) would dominate the MSE.
    clip = {c: tuple(tr[c].quantile(list(winsor))) for c in [TARGET, "growth_lag1"]}
    for f in frames.values():
        for c, (lo, hi) in clip.items():
            f[c] = f[c].clip(lo, hi)

    Xtr = frames["train"][features].to_numpy(float)
    mean = Xtr.mean(axis=0) if standardize else np.zeros(len(features))
    std = Xtr.std(axis=0) if standardize else np.ones(len(features))
    std[std == 0] = 1.0

    def xy(name):
        f = frames[name]
        return (f[features].to_numpy(float) - mean) / std, f[TARGET].to_numpy(float)

    Xtr, ytr = xy("train")
    Xva, yva = xy("val")
    Xte, yte = xy("test")
    Xst, yst = xy("stress")
    return Splits(Xtr, ytr, Xva, yva, Xte, yte, Xst, yst, list(features),
                  mean, std, clip, frames)


if __name__ == "__main__":
    p = build_panel()
    print(p.shape, p.ISO.nunique(), "countries")
    out = os.path.join(DATA_DIR, "processed")
    os.makedirs(out, exist_ok=True)
    p.to_csv(os.path.join(out, "panel.csv"), index=False)
