# Catastrophes climatiques et croissance économique — Projet ML

**Question :** les catastrophes climatiques (inondations, tempêtes, sécheresses, canicules, feux) subies par un pays une année donnée réduisent-elles sa croissance du PIB réel, et de combien ?

**Méthodes implémentées from scratch** (NumPy uniquement) puis comparées à scikit-learn sur des découpages identiques :
- régression linéaire / Ridge : équations normales, descente de gradient, SGD par mini-lots, vérification du gradient, bootstrap par pays ;
- arbre de régression CART : critère SSE, recherche de coupure par sommes cumulées, croissance récursive, importances.

Rapport : [`report/rapport.pdf`](report/rapport.pdf) (source HTML : `report/rapport.html`).

## Données (Kaggle — non redistribuées)
| Jeu | URL |
|---|---|
| EM-DAT, catastrophes naturelles 1900–2021 | https://www.kaggle.com/datasets/brsdincer/all-natural-disasters-19002021-eosdis |
| Banque mondiale, PIB / croissance | https://www.kaggle.com/datasets/zgrcemta/world-gdpgdp-gdp-per-capita-and-annual-growths |

Arborescence attendue après décompression :
```
data/disasters/DISASTERS/1970-2021_DISASTERS.xlsx - emdat data.csv
data/gdp/gdp.csv, gdp_growth.csv, gdp_per_capita.csv, ...
```
Téléchargement, par exemple avec la CLI Kaggle :
```bash
kaggle datasets download -d brsdincer/all-natural-disasters-19002021-eosdis -p data/disasters --unzip
kaggle datasets download -d zgrcemta/world-gdpgdp-gdp-per-capita-and-annual-growths -p data/gdp --unzip
```

## Structure
```
src/
  data_prep.py        construction du panel pays-année, découpage temporel, winsorisation/standardisation (fit sur train)
  scratch_linear.py   régression linéaire/Ridge from scratch (+ check de gradient, pas maximal, bootstrap par pays)
  scratch_tree.py     arbre CART from scratch (+ dépendance partielle)
  plot_style.py       style graphique commun
notebooks/            à exécuter dans l'ordre
  01_exploration_preparation.ipynb
  02_regression_lineaire_scratch.ipynb
  03_arbre_regression_scratch.ipynb
  04_benchmark_sklearn.ipynb     (benchmark séparé : scikit-learn utilisé ici uniquement)
figures/              figures générées (fig01 ... fig20)
results/              tableaux de résultats (CSV)
report/               rapport HTML + PDF
```

## Reproduire
Python 3.13 (testé), graine aléatoire 42 partout.
```bash
pip install -r requirements.txt
```
```bash
cd notebooks
```
```bash
jupyter nbconvert --to notebook --execute --inplace 01_exploration_preparation.ipynb 02_regression_lineaire_scratch.ipynb 03_arbre_regression_scratch.ipynb 04_benchmark_sklearn.ipynb
```
L'ordre compte : le notebook 04 relit les hyperparamètres choisis dans 02 et 03 (`results/`). Durée totale : environ 2 minutes.

## Résultats principaux (test 2013–2019)
| Modèle | RMSE | R² |
|---|---|---|
| Moyenne du train | 3,67 | −0,04 |
| Linéaire from scratch (= sklearn) | **3,13** | **0,244** |
| Arbre CART from scratch (= sklearn, 40 feuilles identiques) | 3,31 | 0,153 |

- Ajouter les catastrophes réduit la RMSE de test de 8,9 % sur les pays frappés (dommages > 0,1 % du PIB).
- Des dommages égaux à 10 % du PIB vont avec une croissance plus faible de 1,5 pt l'année du choc (IC 95 % [−2,5 ; −0,6]), puis un rebond partiel l'année suivante.

## Code externe
Aucun code externe n'est repris. scikit-learn ne sert qu'au benchmark (notebook 04) et aux métriques (MAE, RMSE, R²).
