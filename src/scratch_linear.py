"""Régression linéaire (avec pénalité Ridge optionnelle) implémentée from scratch.

Modèle
------
    ŷ = X w + b        (X : n x p, w : p, b : scalaire)

Fonction objectif (moyenne, régularisation L2 sur w uniquement, pas sur b)
-------------------------------------------------------------------------
    J(w, b) = 1/(2n) * ||X w + b - y||²  +  (alpha/2) * ||w||²

Gradient
--------
    r          = X w + b - y               (résidus)
    dJ/dw      = (1/n) Xᵀ r + alpha * w
    dJ/db      = (1/n) Σ r

Trois procédures d'ajustement sont fournies :
    * "normal" : solution exacte, système (XᵀX/n + alpha*D) θ = Xᵀy/n
                 (θ = [b, w], D = diag(0, 1, ..., 1)) ;
    * "gd"     : descente de gradient à pas fixe (batch complet) ;
    * "sgd"    : descente de gradient stochastique par mini-lots, avec
                 décroissance du pas  eta_k = eta0 / (1 + decay * epoch).

Correspondance avec scikit-learn : Ridge minimise ||y - Xw - b||² + a ||w||²,
donc a_sklearn = n * alpha.

Seules des opérations NumPy génériques (produits matriciels, np.linalg.solve)
sont utilisées ; aucune routine d'apprentissage d'une bibliothèque ML.
"""
from __future__ import annotations

import numpy as np


class LinearRegressionScratch:
    def __init__(self, solver: str = "gd", alpha: float = 0.0, lr: float = 0.1,
                 n_iter: int = 5000, tol: float = 1e-10, batch_size: int = 64,
                 decay: float = 0.01, random_state: int = 42):
        if solver not in ("normal", "gd", "sgd"):
            raise ValueError("solver doit valoir 'normal', 'gd' ou 'sgd'")
        self.solver = solver
        self.alpha = alpha
        self.lr = lr
        self.n_iter = n_iter        # itérations (gd) ou époques (sgd)
        self.tol = tol              # arrêt si |J_k - J_{k-1}| < tol (gd)
        self.batch_size = batch_size
        self.decay = decay
        self.random_state = random_state

    # ----------------------------------------------------------------- objectif
    def loss(self, X, y, w=None, b=None) -> float:
        w = self.coef_ if w is None else w
        b = self.intercept_ if b is None else b
        r = X @ w + b - y
        return 0.5 * np.mean(r ** 2) + 0.5 * self.alpha * np.dot(w, w)

    def gradient(self, X, y, w, b):
        n = X.shape[0]
        r = X @ w + b - y
        return X.T @ r / n + self.alpha * w, r.mean()

    # ------------------------------------------------------------- apprentissage
    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        n, p = X.shape
        self.loss_history_ = []
        if self.solver == "normal":
            self._fit_normal(X, y)
        elif self.solver == "gd":
            self._fit_gd(X, y)
        else:
            self._fit_sgd(X, y)
        return self

    def _fit_normal(self, X, y):
        n, p = X.shape
        Xa = np.hstack([np.ones((n, 1)), X])          # colonne de 1 pour b
        D = np.eye(p + 1)
        D[0, 0] = 0.0                                  # pas de pénalité sur b
        A = Xa.T @ Xa / n + self.alpha * D
        theta = np.linalg.solve(A, Xa.T @ y / n)       # résolution exacte
        self.intercept_, self.coef_ = theta[0], theta[1:]
        self.n_iter_ = 0
        self.loss_history_ = [self.loss(X, y)]

    def _fit_gd(self, X, y):
        n, p = X.shape
        w, b = np.zeros(p), y.mean()     # init : b = moyenne (point de départ raisonnable)
        prev = np.inf
        for k in range(self.n_iter):
            gw, gb = self.gradient(X, y, w, b)
            w -= self.lr * gw
            b -= self.lr * gb
            J = self.loss(X, y, w, b)
            self.loss_history_.append(J)
            if not np.isfinite(J):
                raise FloatingPointError(f"divergence (lr={self.lr} trop grand)")
            if abs(prev - J) < self.tol:  # critère d'arrêt : stagnation de J
                break
            prev = J
        self.coef_, self.intercept_, self.n_iter_ = w, b, k + 1

    def _fit_sgd(self, X, y):
        rng = np.random.default_rng(self.random_state)
        n, p = X.shape
        w, b = np.zeros(p), y.mean()
        for epoch in range(self.n_iter):
            eta = self.lr / (1.0 + self.decay * epoch)
            idx = rng.permutation(n)                      # mélange à chaque époque
            for start in range(0, n, self.batch_size):
                bi = idx[start:start + self.batch_size]
                gw, gb = self.gradient(X[bi], y[bi], w, b)
                w -= eta * gw
                b -= eta * gb
            self.loss_history_.append(self.loss(X, y, w, b))
        self.coef_, self.intercept_, self.n_iter_ = w, b, self.n_iter

    # ----------------------------------------------------------------- inférence
    def predict(self, X):
        return np.asarray(X, float) @ self.coef_ + self.intercept_


# ---------------------------------------------------------------------------
# Vérifications numériques
# ---------------------------------------------------------------------------
def numerical_gradient_check(model: LinearRegressionScratch, X, y, eps=1e-6, seed=0):
    """Compare le gradient analytique à des différences finies centrées.

    Retourne l'erreur relative ||g_ana - g_num|| / (||g_ana|| + ||g_num||).
    """
    rng = np.random.default_rng(seed)
    p = X.shape[1]
    w, b = rng.normal(size=p), rng.normal()
    gw, gb = model.gradient(X, y, w, b)
    g_ana = np.append(gw, gb)
    g_num = np.zeros(p + 1)
    for j in range(p + 1):
        def J(delta):
            ww, bb = w.copy(), b
            if j < p:
                ww[j] += delta
            else:
                bb += delta
            return model.loss(X, y, ww, bb)
        g_num[j] = (J(eps) - J(-eps)) / (2 * eps)
    return np.linalg.norm(g_ana - g_num) / (np.linalg.norm(g_ana) + np.linalg.norm(g_num))


def max_stable_lr(X, alpha=0.0):
    """Pas maximal théorique de la descente de gradient : 2 / L,
    L = plus grande valeur propre de la hessienne (XᵀX/n + alpha I),
    avec la colonne de 1 (biais) incluse."""
    n = X.shape[0]
    Xa = np.hstack([np.ones((n, 1)), X])
    H = Xa.T @ Xa / n
    H[1:, 1:] += alpha * np.eye(X.shape[1])
    return 2.0 / np.linalg.eigvalsh(H).max()


def cluster_bootstrap_coefs(X, y, groups, alpha=0.0, n_boot=500, seed=42):
    """Bootstrap par pays (rééchantillonnage de pays entiers, avec remise).

    Les observations d'un même pays sont corrélées dans le temps ; tirer des
    pays entiers préserve cette dépendance et donne des intervalles de
    confiance plus honnêtes qu'un bootstrap ligne à ligne.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    rows_by_group = {g: np.flatnonzero(groups == g) for g in uniq}
    coefs = np.empty((n_boot, X.shape[1]))
    for i in range(n_boot):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([rows_by_group[g] for g in draw])
        m = LinearRegressionScratch(solver="normal", alpha=alpha).fit(X[idx], y[idx])
        coefs[i] = m.coef_
    return coefs
