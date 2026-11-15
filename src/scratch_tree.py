"""CART regression tree implemented from scratch.

Model
-----
A tree partitions the feature space into rectangular regions R_1..R_M using
binary tests "x_j <= s". The prediction for an observation is the mean of the
target in the leaf (region) it falls into:

    ŷ(x) = Σ_m c_m 1{x ∈ R_m},     c_m = mean(y_i : x_i ∈ R_m)

Impurity criterion: sum of squared errors (SSE), equivalent to the MSE
    SSE(R) = Σ_{i∈R} (y_i - ȳ_R)² = Σ y_i² - (Σ y_i)² / |R|

Best-split search (greedy, recursive)
-------------------------------------
For each feature j, the node's observations are sorted by x_j; cumulative sums
of y and y² then give SSE(left) and SSE(right) for ALL candidate splits in
O(n) after sorting (O(n log n) per feature overall). We keep the (j, s)
maximizing the gain:

    Δ = SSE(parent) - SSE(left) - SSE(right)

The threshold s is the midpoint between two consecutive distinct values (same
convention as scikit-learn). Growth stops when: maximum depth is reached, fewer
than `min_samples_split` observations remain, no split leaves at least
`min_samples_leaf` observations on each side, or the relative gain is below
`min_impurity_decrease`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Node:
    value: float                   # prediction (mean of y in the node)
    n_samples: int
    impurity: float                # MSE of the node
    feature: Optional[int] = None  # None => leaf
    threshold: Optional[float] = None
    left: Optional["Node"] = None
    right: Optional["Node"] = None

    @property
    def is_leaf(self) -> bool:
        return self.feature is None


class RegressionTreeScratch:
    def __init__(self, max_depth: Optional[int] = None, min_samples_split: int = 2,
                 min_samples_leaf: int = 1, min_impurity_decrease: float = 0.0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease

    # ----------------------------------------------------------------- training
    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        self.n_features_ = X.shape[1]
        self._n_total = X.shape[0]
        self._importance = np.zeros(self.n_features_)
        self.root_ = self._grow(X, y, depth=0)
        tot = self._importance.sum()
        self.feature_importances_ = self._importance / tot if tot > 0 else self._importance
        return self

    def _best_split(self, X, y):
        """Returns (gain, feature, threshold) of the best split, or None."""
        n = len(y)
        L = self.min_samples_leaf
        parent_sse = np.sum((y - y.mean()) ** 2)
        best = None
        best_gain = 0.0
        for j in range(X.shape[1]):
            order = np.argsort(X[:, j], kind="mergesort")
            xs, ys = X[order, j], y[order]
            cs, cs2 = np.cumsum(ys), np.cumsum(ys ** 2)
            # split after position i (i = 0..n-2): left = ys[:i+1]
            nl = np.arange(1, n)
            nr = n - nl
            sl, sl2 = cs[:-1], cs2[:-1]
            sr, sr2 = cs[-1] - sl, cs2[-1] - sl2
            sse = (sl2 - sl ** 2 / nl) + (sr2 - sr ** 2 / nr)
            # valid splits: distinct values + minimum leaf size
            valid = (xs[1:] > xs[:-1]) & (nl >= L) & (nr >= L)
            if not valid.any():
                continue
            sse = np.where(valid, sse, np.inf)
            i = int(np.argmin(sse))
            gain = parent_sse - sse[i]
            if gain > best_gain + 1e-12:
                best_gain = gain
                best = (gain, j, 0.5 * (xs[i] + xs[i + 1]))
        return best

    def _grow(self, X, y, depth):
        n = len(y)
        node = Node(value=float(y.mean()), n_samples=n, impurity=float(y.var()))
        if (self.max_depth is not None and depth >= self.max_depth) \
                or n < self.min_samples_split or n < 2 * self.min_samples_leaf \
                or node.impurity == 0.0:
            return node
        split = self._best_split(X, y)
        if split is None:
            return node
        gain, j, s = split
        # same definition as sklearn: weighted impurity decrease
        if gain / self._n_total < self.min_impurity_decrease:
            return node
        mask = X[:, j] <= s
        node.feature, node.threshold = j, s
        self._importance[j] += gain
        node.left = self._grow(X[mask], y[mask], depth + 1)
        node.right = self._grow(X[~mask], y[~mask], depth + 1)
        return node

    # ---------------------------------------------------------------- inference
    def _predict_one(self, x):
        node = self.root_
        while not node.is_leaf:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.value

    def predict(self, X):
        X = np.asarray(X, float)
        return np.array([self._predict_one(x) for x in X])

    # ------------------------------------------------------------------ helpers
    def depth(self, node=None):
        node = self.root_ if node is None else node
        return 0 if node.is_leaf else 1 + max(self.depth(node.left), self.depth(node.right))

    def n_leaves(self, node=None):
        node = self.root_ if node is None else node
        return 1 if node.is_leaf else self.n_leaves(node.left) + self.n_leaves(node.right)

    def to_text(self, feature_names=None, max_depth=3, unscale=None, node=None, depth=0):
        """Text rendering of the top levels of the tree.
        `unscale(j, s)` shows thresholds in their original units."""
        node = self.root_ if node is None else node
        pad = "|   " * depth
        if node.is_leaf or depth >= max_depth:
            return f"{pad}-> ŷ = {node.value:.2f}  (n={node.n_samples})\n"
        name = feature_names[node.feature] if feature_names else f"x{node.feature}"
        s = unscale(node.feature, node.threshold) if unscale else node.threshold
        out = f"{pad}{name} <= {s:.3f}  (n={node.n_samples})\n"
        out += self.to_text(feature_names, max_depth, unscale, node.left, depth + 1)
        out += f"{pad}{name} >  {s:.3f}\n"
        out += self.to_text(feature_names, max_depth, unscale, node.right, depth + 1)
        return out


def partial_dependence(model, X, j, grid):
    """Partial dependence: mean prediction when x_j is set to each value of
    `grid` (the other features keep their observed values)."""
    out = []
    for v in grid:
        Xm = X.copy()
        Xm[:, j] = v
        out.append(model.predict(Xm).mean())
    return np.array(out)
