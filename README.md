# Climate Disasters and Economic Growth — ML Project

**Question:** do the climate disasters (floods, storms, droughts, heatwaves, wildfires) that hit a country in a given year reduce its real GDP growth, and by how much?

**Methods implemented from scratch** (NumPy only), then compared with scikit-learn on identical splits:
- linear / Ridge regression: normal equations, gradient descent, mini-batch SGD, gradient check, country-level bootstrap;
- CART regression tree: SSE criterion, split search with cumulative sums, recursive growth, feature importances.

Report: [`report/report.pdf`](report/report.pdf) (HTML source: `report/report.html`).

## Data (Kaggle)
| Dataset | URL |
|---|---|
| EM-DAT, natural disasters 1900–2021 | https://www.kaggle.com/datasets/brsdincer/all-natural-disasters-19002021-eosdis |
| World Bank, GDP / growth | https://www.kaggle.com/datasets/zgrcemta/world-gdpgdp-gdp-per-capita-and-annual-growths |

Expected layout after unzipping:
```
data/disasters/DISASTERS/1970-2021_DISASTERS.xlsx - emdat data.csv
data/gdp/gdp.csv, gdp_growth.csv, gdp_per_capita.csv, ...
```
To download, for example with the Kaggle CLI:
```bash
kaggle datasets download -d brsdincer/all-natural-disasters-19002021-eosdis -p data/disasters --unzip
kaggle datasets download -d zgrcemta/world-gdpgdp-gdp-per-capita-and-annual-growths -p data/gdp --unzip
```

## Structure
```
src/
  data_prep.py        country-year panel, chronological split, winsorization/standardization (fitted on train)
  scratch_linear.py   linear/Ridge regression from scratch (+ gradient check, maximum step, country bootstrap)
  scratch_tree.py     CART tree from scratch (+ partial dependence)
  plot_style.py       shared plotting style
notebooks/            run in order
  01_exploration_preparation.ipynb
  02_linear_regression_scratch.ipynb
  03_regression_tree_scratch.ipynb
  04_benchmark_sklearn.ipynb     (separate benchmark: scikit-learn is only used here)
figures/              generated figures (fig01 ... fig20)
results/              result tables (CSV)
report/               HTML + PDF report
```

## Reproduce
Python 3.13 (tested), random seed 42 everywhere.
```bash
pip install -r requirements.txt
```
```bash
cd notebooks
```
```bash
jupyter nbconvert --to notebook --execute --inplace 01_exploration_preparation.ipynb 02_linear_regression_scratch.ipynb 03_regression_tree_scratch.ipynb 04_benchmark_sklearn.ipynb
```
Order matters: notebook 04 reads the hyperparameters selected in 02 and 03 (`results/`). Total run time: about 2 minutes.

## Main results (test 2013–2019)
| Model | RMSE | R² |
|---|---|---|
| Training mean | 3.67 | −0.04 |
| Linear from scratch (= sklearn) | **3.13** | **0.244** |
| CART tree from scratch (= sklearn, 40 identical leaves) | 3.31 | 0.153 |

- Adding the disasters lowers the test RMSE by 8.9 % for the countries hit (damages > 0.1 % of GDP).
- Damages equal to 10 % of GDP go with growth 1.5 pt lower in the shock year (95 % CI [−2.5, −0.6]), then a partial rebound the following year.

## External code and AI assistance
No external code is reused. scikit-learn is only used for the benchmark (notebook 04) and the metrics (MAE, RMSE, R²).

This project was developed with the help of an AI assistant (Claude, Anthropic) for code, report and slides; we reviewed, ran and understood all of it.
