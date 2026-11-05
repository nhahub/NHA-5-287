"""Model building blocks added in v5. Deliberately self-contained (numpy / scikit-learn only): a pickled model references the module that defines its class, so the
exported prediction package vendors this file.

Time series
  SeasonalNaive            predict the value one season ago (the bar every forecaster must clear on seasonal data)
  TrendSeasonalRegressor   log/linear trend (degree 1-2) + per-season effects, fitted by ridge; extrapolates a trend and a season, which trees cannot
  SeasonalDeltaRegressor   wraps ANY regressor so it learns the CHANGE versus one season ago (a ratio on the log scale), then adds the season back.
                           Makes tree / boosting models usable on trending, seasonal series.
Neural networks (small multi-layer perceptrons, early stopping, scaled targets) for tabular data
  MLPRegressorNet / MLPClassifierNet
"""
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone


class SeasonalNaive(BaseEstimator, RegressorMixin):
    """Forecast = the observation one season ago, read from the lag column `col`."""

    def __init__(self, col: str = ""):
        self.col = col

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return X[self.col].to_numpy(dtype=float)


class TrendSeasonalRegressor(BaseEstimator, RegressorMixin):
    """y ~ trend(t) + season(phase). Needs the columns `t_col` (position in the series) and `phase_col` (position modulo the period) that the time-series
    preparation adds. `log=True` models the multiplicative case (seasonal swings that grow with the level). Falls back to the raw scale when y <= 0."""

    def __init__(self, period: int = 12, degree: int = 1, alpha: float = 1e-3, log: bool = True, t_col: str = "ts_t", phase_col: str = "ts_phase"):
        self.period, self.degree, self.alpha, self.log = period, degree, alpha, log
        self.t_col, self.phase_col = t_col, phase_col

    def _design(self, X):
        t = (X[self.t_col].to_numpy(dtype=float) - self.t0_) / self.ts_
        ph = X[self.phase_col].to_numpy(dtype=float).astype(int) % max(int(self.period), 1)
        cols = [np.ones_like(t)] + [t ** d for d in range(1, int(self.degree) + 1)]
        p = int(self.period)
        if p > 1:
            if p <= 24:                                            # one dummy per season (first season is the reference)
                cols += [(ph == k).astype(float) for k in range(1, p)]
            else:                                                  # long seasons: a few Fourier terms instead of 50+ dummies
                for k in range(1, min(6, p // 2) + 1):
                    cols += [np.sin(2 * np.pi * k * ph / p), np.cos(2 * np.pi * k * ph / p)]
        return np.column_stack(cols)

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        t = X[self.t_col].to_numpy(dtype=float)
        self.t0_, self.ts_ = float(t.min()), float(max(t.max() - t.min(), 1.0))
        self.use_log_ = bool(self.log) and bool((y > 0).all())
        z = np.log(y) if self.use_log_ else y
        D = self._design(X)
        pen = np.eye(D.shape[1]) * float(self.alpha)
        pen[0, 0] = 0.0                                            # never shrink the intercept
        self.coef_ = np.linalg.solve(D.T @ D + pen, D.T @ z)
        return self

    def predict(self, X):
        z = self._design(X) @ self.coef_
        return np.exp(z) if self.use_log_ else z


class SeasonalDeltaRegressor(BaseEstimator, RegressorMixin):
    """Fit `regressor` on the change versus the reference lag column `col` (one season ago, or the previous step when there is no season), then add it back.
    log=True works on log ratios, which is right when seasonal swings scale with the level."""

    def __init__(self, regressor=None, col: str = "", log: bool = True):
        self.regressor, self.col, self.log = regressor, col, log

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        ref = X[self.col].to_numpy(dtype=float)
        self.use_log_ = bool(self.log) and bool((y > 0).all()) and bool((ref > 0).all())
        z = (np.log(y) - np.log(ref)) if self.use_log_ else (y - ref)
        self.regressor_ = clone(self.regressor).fit(X, z)
        return self

    def predict(self, X):
        ref = X[self.col].to_numpy(dtype=float)
        d = np.asarray(self.regressor_.predict(X), dtype=float)
        return np.exp(d + np.log(np.where(ref > 0, ref, 1e-12))) if self.use_log_ else d + ref


def _layers(h1: int, h2: int):
    return (int(h1),) if int(h2) <= 0 else (int(h1), int(h2))


class MLPRegressorNet(BaseEstimator, RegressorMixin):
    """Multi-layer perceptron for regression. The target is standardised internally (a raw-scale target makes an MLP train badly), early stopping on a
    validation slice, several restarts averaged only if requested. Inputs are expected to be scaled (the pipeline's preprocessor does that)."""

    def __init__(self, h1: int = 64, h2: int = 32, alpha: float = 1e-3, learning_rate_init: float = 1e-3, max_iter: int = 600, n_models: int = 1, random_state: Optional[int] = 42):
        self.h1, self.h2, self.alpha, self.learning_rate_init = h1, h2, alpha, learning_rate_init
        self.max_iter, self.n_models, self.random_state = max_iter, n_models, random_state

    def fit(self, X, y):
        from sklearn.neural_network import MLPRegressor
        y = np.asarray(y, dtype=float)
        self.mu_, self.sd_ = float(y.mean()), float(y.std() or 1.0)
        z = (y - self.mu_) / self.sd_
        self.models_ = [MLPRegressor(hidden_layer_sizes=_layers(self.h1, self.h2), alpha=self.alpha, learning_rate_init=self.learning_rate_init, max_iter=self.max_iter,
                                     early_stopping=len(y) >= 200, validation_fraction=0.15, n_iter_no_change=25, random_state=(self.random_state or 0) + i).fit(X, z)
                        for i in range(int(self.n_models))]
        return self

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.models_], axis=0) * self.sd_ + self.mu_


class MLPClassifierNet(BaseEstimator, ClassifierMixin):
    def __init__(self, h1: int = 64, h2: int = 32, alpha: float = 1e-3, learning_rate_init: float = 1e-3, max_iter: int = 600, random_state: Optional[int] = 42):
        self.h1, self.h2, self.alpha, self.learning_rate_init = h1, h2, alpha, learning_rate_init
        self.max_iter, self.random_state = max_iter, random_state

    def fit(self, X, y):
        from sklearn.neural_network import MLPClassifier
        self.model_ = MLPClassifier(hidden_layer_sizes=_layers(self.h1, self.h2), alpha=self.alpha, learning_rate_init=self.learning_rate_init, max_iter=self.max_iter,
                                    early_stopping=len(y) >= 200, validation_fraction=0.15, n_iter_no_change=25, random_state=self.random_state).fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return self.model_.predict(X)
