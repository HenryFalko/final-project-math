"""Construction du panel pays-année à partir des deux jeux de données Kaggle.

Sources
-------
* EM-DAT (via Kaggle, brsdincer/all-natural-disasters-19002021-eosdis) :
  une ligne = une catastrophe naturelle (pays, année, type, morts, sinistrés, dommages).
* Banque mondiale (via Kaggle, zgrcemta/world-gdpgdp-gdp-per-capita-and-annual-growths) :
  PIB courant (USD), PIB par habitant, croissance annuelle du PIB réel (%).

Unité d'observation : (pays, année) entre 1971 et 2020.
Cible : croissance annuelle du PIB réel (%) l'année t.

Toutes les transformations qui « apprennent » quelque chose des données
(bornes de winsorisation, moyennes/écarts-types de standardisation) sont
estimées sur l'échantillon d'entraînement uniquement -> pas de fuite de données.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")

# Catastrophes d'origine climatique (classification EM-DAT : hydrologique,
# météorologique, climatologique). Les séismes / volcans (géophysiques) servent
# de groupe de comparaison : ils ne sont pas liés au climat.
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

# Variables explicatives finales (dans cet ordre).
DISASTER_FEATURES = [
    "log_dmg_climate_pct",       # dommages climatiques en % du PIB (log1p), année t
    "log_dmg_climate_pct_lag1",  # idem, année t-1 (effet retardé)
    "log_deaths_climate_pm",     # morts par million d'habitants (log1p)
    "log_affected_climate_pct",  # sinistrés en % de la population (log1p)
    "n_hydro",                   # nombre d'inondations / glissements
    "n_meteo",                   # nombre de tempêtes
    "n_climato",                 # nombre de sécheresses / canicules / feux
    "log_dmg_geo_pct",           # dommages géophysiques (contrôle / comparaison)
]
CONTROL_FEATURES = [
    "growth_lag1",     # croissance de l'année précédente (inertie)
    "log_gdppc_lag1",  # niveau de développement (log PIB/hab t-1)
    "world_growth",    # cycle économique mondial l'année t
]
FEATURES = DISASTER_FEATURES + CONTROL_FEATURES
TARGET = "gdp_growth"

# Découpage temporel (pas de mélange passé / futur).
TRAIN_YEARS = (1971, 2005)
VAL_YEARS = (2006, 2012)
TEST_YEARS = (2013, 2019)
STRESS_YEARS = (2020, 2020)  # année COVID, analysée à part


# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------
def load_disasters(data_dir: str = DATA_DIR) -> pd.DataFrame:
    path = glob.glob(os.path.join(data_dir, "disasters", "DISASTERS", "1970-2021*"))[0]
    return pd.read_csv(path)


def _wb_long(name: str, data_dir: str = DATA_DIR) -> pd.DataFrame:
    """Fichier Banque mondiale (format large) -> format long (ISO, year, value)."""
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
# Agrégation des catastrophes par pays-année
# ---------------------------------------------------------------------------
def aggregate_disasters(dis: pd.DataFrame) -> pd.DataFrame:
    d = dis.copy()
    d["group"] = d["Disaster Type"].map(CLIMATE_GROUPS)
    d.loc[d["Disaster Type"].isin(GEO_TYPES), "group"] = "geo"
    d = d[d["group"].notna()]
    d["is_climate"] = d["group"] != "geo"
    # Valeur manquante = non renseignée dans EM-DAT ; on la compte comme 0
    # (hypothèse discutée dans le rapport : biais de déclaration).
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
# Panel complet
# ---------------------------------------------------------------------------
def build_panel(data_dir: str = DATA_DIR) -> pd.DataFrame:
    dis = load_disasters(data_dir)
    wb = load_world_bank(data_dir)

    world = wb.loc[wb.ISO == "WLD", ["year", "gdp_growth"]].rename(
        columns={"gdp_growth": "world_growth"})

    # Univers de pays = pays présents dans EM-DAT (exclut les agrégats WB
    # comme "Euro area", "World"...).
    countries = set(dis["ISO"].unique())
    wb = wb[wb.ISO.isin(countries)].copy()

    agg = aggregate_disasters(dis)
    panel = wb.merge(agg, on=["ISO", "year"], how="left")
    fill_cols = [c for c in agg.columns if c not in ("ISO", "year")]
    panel[fill_cols] = panel[fill_cols].fillna(0.0)  # aucune catastrophe enregistrée
    panel = panel.merge(world, on="year", how="left")

    panel["population"] = panel["gdp"] / panel["gdp_per_capita"]
    # Dommages (milliers USD courants) rapportés au PIB (USD courants) -> %.
    panel["dmg_climate_pct"] = 100 * panel["dmg_climate"] * 1e3 / panel["gdp"]
    panel["dmg_geo_pct"] = 100 * panel["dmg_geo"] * 1e3 / panel["gdp"]
    panel["deaths_climate_pm"] = 1e6 * panel["deaths_climate"] / panel["population"]
    panel["affected_climate_pct"] = 100 * panel["affected_climate"] / panel["population"]

    for c in ["dmg_climate_pct", "dmg_geo_pct", "deaths_climate_pm", "affected_climate_pct"]:
        panel[f"log_{c}"] = np.log1p(panel[c])

    panel = panel.sort_values(["ISO", "year"])
    g = panel.groupby("ISO")
    # Décalages : on vérifie que l'année précédente est bien t-1.
    prev_year = g["year"].shift(1)
    ok = prev_year == panel["year"] - 1
    panel["growth_lag1"] = g["gdp_growth"].shift(1).where(ok)
    panel["log_gdppc_lag1"] = np.log(g["gdp_per_capita"].shift(1)).where(ok)
    panel["log_dmg_climate_pct_lag1"] = g["log_dmg_climate_pct"].shift(1).where(ok)

    panel = panel[(panel.year >= 1971) & (panel.year <= 2020)]
    panel = panel.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)
    return panel


# ---------------------------------------------------------------------------
# Découpage + pré-traitements ajustés sur le train
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
    frames: dict  # DataFrames bruts par split (pour l'analyse)


def make_splits(panel: pd.DataFrame, features=FEATURES, standardize=True,
                winsor=(0.01, 0.99)) -> Splits:
    """Découpage temporel + winsorisation + standardisation (paramètres = train)."""
    frames = {
        "train": panel[year_mask(panel, TRAIN_YEARS)].copy(),
        "val": panel[year_mask(panel, VAL_YEARS)].copy(),
        "test": panel[year_mask(panel, TEST_YEARS)].copy(),
        "stress": panel[year_mask(panel, STRESS_YEARS)].copy(),
    }
    tr = frames["train"]
    # Bornes de winsorisation (cible + croissance retardée) estimées sur le train :
    # quelques valeurs extrêmes (guerres, ex. Irak, Libye : +/-60 %) écraseraient la MSE.
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
    print(p.shape, p.ISO.nunique(), "pays")
    out = os.path.join(DATA_DIR, "processed")
    os.makedirs(out, exist_ok=True)
    p.to_csv(os.path.join(out, "panel.csv"), index=False)
