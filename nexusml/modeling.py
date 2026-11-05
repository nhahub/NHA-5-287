"""Modeling tools. Every candidate is a full sklearn Pipeline (preprocessing + model), so
preprocessing is fit only inside CV folds / on train, and the saved model scores raw data.

Honest selection
  * Default models are scored by k-fold CV on the TRAIN split.
  * Hyper-parameter tuning uses RandomizedSearchCV *inside* an outer CV (nested CV). The nested
    score is an unbiased estimate of "tune, then deploy"; it (not the search's own best score)
    is what competes against the default models. Tuning only wins if it helps out-of-fold.
  * The held-out test set is scored once per candidate and never drives selection.
"""
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (AdaBoostClassifier, AdaBoostRegressor, ExtraTreesClassifier, ExtraTreesRegressor,
                              HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, davies_bouldin_score, f1_score,
                             mean_absolute_error, mean_squared_error, r2_score, roc_auc_score,
                             silhouette_score)
from sklearn.model_selection import (KFold, RandomizedSearchCV, StratifiedKFold, TimeSeriesSplit,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, OneHotEncoder, StandardScaler

from . import config, estimators as est_mod, imbalance, threshold as thr_mod
from .profiling import to_datetime
from .trends import choose_freq

DEFAULT_STRATEGY = {"class_weight": None, "tune": True, "extra_models": False, "log_target": False,
                    "delta_target": False, "tune_n_iter": config.TUNE_N_ITER,
                    "imbalance": None,                    # None | "smote" | "adasyn" | "random": resample TRAINING folds only (see imbalance.py)
                    "tune_threshold": True,               # binary classification: pick the decision threshold from out-of-fold PR/F1 instead of 0.5
                    "threshold_objective": "f1"}          # f1 | fbeta | pr_distance | youden | cost


class LastValueRegressor(BaseEstimator, RegressorMixin):
    """Naive forecasting baseline: predict the previous observation (the `*_lag1` column)."""

    def __init__(self, col: str = ""):
        self.col = col

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return X[self.col].to_numpy(dtype=float)


class DeltaRegressor(BaseEstimator, RegressorMixin):
    """Learns the CHANGE vs the previous value, then adds it back. Lets tree models forecast a
    trending series (trees cannot extrapolate levels, but changes are roughly stationary)."""

    def __init__(self, regressor=None, col: str = ""):
        self.regressor = regressor
        self.col = col

    def fit(self, X, y):
        self.regressor_ = clone(self.regressor)
        self.regressor_.fit(X, np.asarray(y, dtype=float) - X[self.col].to_numpy(dtype=float))
        return self

    def predict(self, X):
        return self.regressor_.predict(X) + X[self.col].to_numpy(dtype=float)
SUPERVISED = ("classification", "regression", "time_series")


class Spec:
    """A candidate = an estimator + HOW its input is represented (text representation, image feature backend, whether tabular columns are kept).
    Plain estimators (the v4 candidates) are still accepted everywhere; a Spec is only used by the text / image suites."""

    def __init__(self, model, text_rep: Optional[str] = None, image_rep: Optional[str] = None, meta: bool = True, label: str = "",
                 recipe: Tuple[str, ...] = (), ideas: Optional[List[Dict[str, Any]]] = None, ts: Optional[Dict[str, Any]] = None):
        self.model, self.text_rep, self.image_rep, self.meta, self.label = model, text_rep, image_rep, meta, label
        self.ts = ts                                                           # time-series family descriptor: {"kind": naive|seasonal_naive|trend_seasonal|delta, ...}
        self.recipe, self.ideas = tuple(recipe or ()), ideas or None            # AutoML feature recipes (automl_recipes.py)


def _unspec(m):
    return (m.model, m) if isinstance(m, Spec) else (m, None)


def space_key(name: str) -> str:
    """'LogisticRegression · char TF-IDF' -> 'LogisticRegression' (the key into the search-space tables)."""
    return str(name).split(" · ")[0]


def primary_metric_for(task: str, n_classes: int = 2) -> str:
    if task == "classification":
        return "roc_auc" if n_classes == 2 else "f1_macro"
    return "r2"


def _scoring(task: str, n_classes: int = 2) -> str:
    return primary_metric_for(task, n_classes)


# data
def expand_datetimes(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            continue
        dt = to_datetime(df[c])
        df[f"{c}_year"] = dt.dt.year
        df[f"{c}_month"] = dt.dt.month
        df[f"{c}_day"] = dt.dt.day
        df[f"{c}_dayofweek"] = dt.dt.dayofweek
        df[f"{c}_ordinal"] = (dt - pd.Timestamp("1970-01-01")).dt.days
        df = df.drop(columns=[c])
    return df


def clean_frame(df: pd.DataFrame, drop_cols: List[str], datetime_cols: List[str]) -> pd.DataFrame:
    out = df.drop(columns=[c for c in drop_cols if c in df.columns])
    out = expand_datetimes(out, datetime_cols)
    for c in out.columns:
        s = out[c]
        if pd.api.types.is_bool_dtype(s):
            out[c] = s.astype(str)
        elif not pd.api.types.is_numeric_dtype(s):
            o = s.astype(object)
            out[c] = o.where(o.notna(), np.nan)
    return out


def default_lags(dates: pd.Series) -> Tuple[List[int], List[int]]:
    d = pd.Series(dates).dropna().sort_values().drop_duplicates()
    step = d.diff().dt.days.median() if len(d) > 2 else 1
    if step >= 25:
        return [1, 2, 3], [3]
    if step >= 6:
        return [1, 2, 4], [4]
    return [1, 2, 7], [7]


def add_lag_features(d: pd.DataFrame, target: str, lags: List[int], rolls: List[int]) -> pd.DataFrame:
    """Lags/rolling means use ONLY past values (shift >= 1), so no look-ahead leakage."""
    d = d.copy()
    for k in lags:
        d[f"{target}_lag{k}"] = d[target].shift(k)
    for w in rolls:
        d[f"{target}_roll{w}"] = d[target].shift(1).rolling(w).mean()
    return d


def prepare_timeseries(df: pd.DataFrame, plan: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    date, target = plan["date_column"], plan["target"]
    d = df.drop(columns=[c for c in plan.get("drop_columns", []) if c in df.columns and c not in (date, target)]).copy()
    d[date] = to_datetime(d[date])
    d = d.dropna(subset=[date, target]).sort_values(date, kind="stable")
    info: Dict[str, Any] = {"aggregated": False}
    if d[date].duplicated().mean() > 0.1:            # transaction-level rows -> one row per period
        freq = plan.get("freq") or choose_freq(d[date])
        agg: Dict[str, Any] = {}
        for c in d.columns:
            if c == date:
                continue
            if c == target:
                agg[c] = plan.get("target_agg", "sum")
            elif pd.api.types.is_numeric_dtype(d[c]) and not pd.api.types.is_bool_dtype(d[c]):
                agg[c] = "mean"
            else:
                agg[c] = lambda s: s.mode().iloc[0] if len(s.dropna()) else np.nan
        d = d.groupby(pd.Grouper(key=date, freq=freq)).agg(agg).dropna(subset=[target]).reset_index()
        info.update(aggregated=True, freq=freq, target_agg=plan.get("target_agg", "sum"))
    from . import advisor
    d = d.reset_index(drop=True)
    cut = max(int(len(d) * (1 - config.TEST_SIZE)), 12)                       # season / trend are measured on the TRAINING part only (no peeking at the test window)
    ins = advisor.series_insights(d[target].to_numpy(dtype=float)[:cut], d[date].iloc[:cut])
    lags, rolls = plan.get("lags"), plan.get("rolls")
    if not lags:
        lags, rolls = advisor.ts_lags(ins, *default_lags(d[date]), n=len(d))
    per = int(ins.get("period") or 0)
    pos = np.arange(len(d), dtype=float)
    d["ts_t"] = pos                                                               # position in the series: lets a model extrapolate trend
    d["ts_phase"] = pos % max(per, 1)                                              # position inside the season
    if per:
        d["ts_sin"], d["ts_cos"] = np.sin(2 * np.pi * pos / per), np.cos(2 * np.pi * pos / per)
    info["insights"] = ins
    d = add_lag_features(d, target, lags, rolls)
    lagcols = [c for c in d.columns if c.startswith(f"{target}_lag") or c.startswith(f"{target}_roll")]
    d = d.dropna(subset=lagcols).reset_index(drop=True)
    info.update(lags=lags, rolls=rolls)
    return d, info


def make_splits(df: pd.DataFrame, plan: Dict[str, Any]) -> Dict[str, Any]:
    pt, target = plan["problem_type"], plan.get("target")
    if plan.get("image_root"):
        from . import featurizers
        featurizers.set_image_root(plan["image_root"])
        featurizers.set_image_prep(plan.get("image_prep"))
        featurizers.set_deep_mode(plan.get("deep_mode"))
        for col in df.columns:
            if featurizers.is_image_path(df[col]):
                featurizers.check_images(df[col], plan["image_root"])
    if pt not in SUPERVISED:                          # clustering / descriptive: no target, no split
        X = clean_frame(df, plan["drop_columns"], plan["datetime_columns"])
        return {"X_train": X, "X_test": None, "y_train": None, "y_test": None, "classes": None,
                "summary": {"n_rows_used": int(len(X)), "n_train": int(len(X)), "n_test": 0,
                            "n_features": int(X.shape[1]), "split": "none"}, "ts_info": None}
    ts_info = None
    if pt == "time_series":
        data, ts_info = prepare_timeseries(df, plan)
    else:
        data = df.dropna(subset=[target]).copy()
    y_raw = data[target]
    X = clean_frame(data.drop(columns=[target]), plan["drop_columns"], plan["datetime_columns"])

    classes: Optional[List[str]] = None
    positive = None
    if pt == "classification":
        s = y_raw.astype(str)
        counts = s.value_counts()
        if len(counts) == 2:                          # minority class = event of interest = code 1
            classes = [str(counts.index[0]), str(counts.index[1])]
            positive = classes[1]
        else:
            classes = sorted(counts.index.tolist())
        y = s.map({c: i for i, c in enumerate(classes)}).to_numpy()
    else:
        y = y_raw.astype(float).to_numpy()

    n = len(X)
    if pt == "time_series":
        cut = int(n * (1 - config.TEST_SIZE))
        X_train, X_test, y_train, y_test = X.iloc[:cut], X.iloc[cut:], y[:cut], y[cut:]
    else:
        stratify = None
        if pt == "classification":
            cnt = np.bincount(y)
            if cnt.min() >= 2 and len(cnt) > 1:
                stratify = y
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=stratify)

    summary: Dict[str, Any] = {
        "n_rows_used": int(n), "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "n_features": int(X.shape[1]), "split": "chronological" if pt == "time_series" else "random"}
    if ts_info:
        summary["time_series"] = ts_info
    if pt == "classification":
        cnt = np.bincount(y_train, minlength=len(classes))
        summary["class_distribution"] = {c: int(k) for c, k in zip(classes, cnt)}
        summary["minority_share"] = float(cnt[cnt > 0].min() / cnt.sum())
        summary["positive_class"] = positive
    else:
        yt = pd.Series(y_train)
        summary.update(target_skew=float(yt.skew()) if len(yt) > 2 else 0.0,
                       target_min=float(yt.min()), target_mean=float(yt.mean()))
    return {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
            "classes": classes, "positive_class": positive, "summary": summary, "ts_info": ts_info}


# models
def build_preprocessor(X: pd.DataFrame, spec: Optional["Spec"] = None) -> ColumnTransformer:
    """Tabular columns -> impute/scale/one-hot; text columns -> a text representation; image-path columns -> an image feature backend.
    Without a `spec` this is exactly the v4 behaviour (dense 300-feature TF-IDF, handcrafted image features)."""
    from . import featurizers
    mods = featurizers.modalities(X)
    special = set(mods["text"]) | set(mods["image"])
    num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat = [c for c in X.columns if c not in num and c not in special]
    sparse_out = False
    if spec is not None and spec.text_rep:
        from . import nlp
        extra = [(f"text_{i}", nlp.text_transformer(spec.text_rep), c) for i, c in enumerate(mods["text"])]
        sparse_out = nlp.is_sparse_rep(spec.text_rep)
    else:
        extra = [(f"text_{i}", featurizers.TextFeaturizer(), c) for i, c in enumerate(mods["text"])]
    backend = spec.image_rep if (spec is not None and spec.image_rep) else None
    extra += [(f"img_{i}", Pipeline([("f", featurizers.ImageFeaturizer(backend=backend)), ("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
                                     ("sc", StandardScaler())]), [c]) for i, c in enumerate(mods["image"])]
    keep_meta = spec.meta if spec is not None else True
    recipe = tuple(getattr(spec, "recipe", ()) or ())
    tab = []
    if keep_meta:
        num_steps = [("imp", SimpleImputer(strategy="median", add_indicator="missing_ind" in recipe))]
        num_steps.append(("pt", PowerTransformer(method="yeo-johnson", standardize=True)) if "power" in recipe else ("sc", StandardScaler()))
        hi = [c for c in cat if X[c].nunique() > 12] if "target_enc" in recipe else []
        lo = [c for c in cat if c not in hi]
        tab = [("num", Pipeline(num_steps), num),
               ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                 ("ohe", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01,
                                                       max_categories=30, sparse_output=False))]), lo)]
        if hi:
            from sklearn.preprocessing import TargetEncoder
            tab.append(("tenc", TargetEncoder(random_state=config.RANDOM_STATE), hi))
    return ColumnTransformer(extra + tab, remainder="drop", sparse_threshold=0.3 if sparse_out else 0)


def _optional_boosters(task: str, rs: int) -> Dict[str, Any]:
    """XGBoost / LightGBM only when they happen to be installed (never required)."""
    out: Dict[str, Any] = {}
    try:
        import xgboost as xgb
        out["XGBoost"] = (xgb.XGBClassifier(n_estimators=200, learning_rate=0.1, eval_metric="logloss", random_state=rs, n_jobs=1, verbosity=0) if task == "classification"
                          else xgb.XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=rs, n_jobs=1, verbosity=0))
    except Exception:
        pass
    try:
        import lightgbm as lgb
        out["LightGBM"] = (lgb.LGBMClassifier(n_estimators=200, random_state=rs, n_jobs=1, verbose=-1) if task == "classification"
                           else lgb.LGBMRegressor(n_estimators=200, random_state=rs, n_jobs=1, verbose=-1))
    except Exception:
        pass
    return out


def enrich_strategy(task: str, splits: Dict[str, Any], strategy: Dict[str, Any]) -> Dict[str, Any]:
    """Copy what the time-series preparation learned (season length, reference lag column, log scale) into the strategy so the candidate builder can create
    the seasonal families. Plain primitives only (the strategy is hashed and exported)."""
    out = dict(strategy)
    ti = splits.get("ts_info") or {}
    if task == "time_series" and ti.get("insights") is not None:
        X, ins = splits["X_train"], ti["insights"]
        per = int(ins.get("period") or 0)
        lag1 = lag1_col(X)
        ref = next((c for c in X.columns if per and str(c).endswith(f"_lag{per}")), lag1)
        if lag1 and "ts_t" in X.columns:
            out.update(ts_period=per, ts_ref=ref, ts_lag1=lag1, ts_log=bool(ins.get("log_scale")))
    return out


def _ts_families(strategy: Dict[str, Any], rs: int) -> Dict[str, Any]:
    per, ref, lag1, lg = int(strategy.get("ts_period") or 0), strategy["ts_ref"], strategy["ts_lag1"], bool(strategy.get("ts_log"))
    delta = lambda m: Spec(m, ts={"kind": "delta", "ref": ref, "log": lg})
    return {"Naive (last value)": Spec(est_mod.SeasonalNaive(col=lag1), ts={"kind": "naive", "lag1": lag1}),
            **({"Seasonal naive": Spec(est_mod.SeasonalNaive(col=ref), ts={"kind": "seasonal_naive", "ref": ref})} if per else {}),
            "TrendSeasonal": Spec(est_mod.TrendSeasonalRegressor(period=max(per, 1), degree=1, log=lg), ts={"kind": "trend_seasonal"}),
            "Ridge · seasonal Δ": delta(Ridge(alpha=1.0)),
            "HistGradientBoosting · seasonal Δ": delta(HistGradientBoostingRegressor(random_state=rs, max_iter=200, learning_rate=0.05, max_depth=3))}


def _candidates(task: str, strategy: Dict[str, Any]) -> Dict[str, Any]:
    kind = strategy.get("pipeline_kind")
    if kind == "text" and task in ("classification", "regression"):
        from . import nlp
        return nlp.candidate_specs(task, strategy, config.RANDOM_STATE)
    if kind == "image" and task in ("classification", "regression"):
        from . import cv
        return cv.candidate_specs(task, strategy, config.RANDOM_STATE)
    rs, cw = config.RANDOM_STATE, strategy.get("class_weight")
    rg = bool(strategy.get("regularize"))
    if task == "classification":
        m = {
            "LogisticRegression": LogisticRegression(max_iter=2000, class_weight=cw, C=0.1 if rg else 1.0),
            "DecisionTree": DecisionTreeClassifier(max_depth=4 if rg else 6, min_samples_leaf=10 if rg else 1, class_weight=cw, random_state=rs),
            "KNN": KNeighborsClassifier(n_neighbors=25 if rg else 15),
            "RandomForest": RandomForestClassifier(n_estimators=200, class_weight=cw, n_jobs=-1, random_state=rs,
                                                   **({"min_samples_leaf": 5, "max_depth": 10} if rg else {})),
            "HistGradientBoosting": HistGradientBoostingClassifier(class_weight=cw, random_state=rs,
                                                                   **({"l2_regularization": 1.0, "max_depth": 4} if rg else {})),
        }
        if strategy.get("extra_models"):
            m["ExtraTrees"] = ExtraTreesClassifier(n_estimators=300, class_weight=cw, n_jobs=-1, random_state=rs)
            m["AdaBoost"] = AdaBoostClassifier(n_estimators=150, random_state=rs)
    else:
        m = {
            "Ridge": Ridge(alpha=10.0 if rg else 1.0),
            "DecisionTree": DecisionTreeRegressor(max_depth=4 if rg else 6, min_samples_leaf=10 if rg else 1, random_state=rs),
            "KNN": KNeighborsRegressor(n_neighbors=25 if rg else 15),
            "RandomForest": RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=rs, **({"min_samples_leaf": 5, "max_depth": 10} if rg else {})),
            "HistGradientBoosting": HistGradientBoostingRegressor(random_state=rs, **({"l2_regularization": 1.0, "max_depth": 4} if rg else {})),
        }
        if strategy.get("extra_models"):
            m["ExtraTrees"] = ExtraTreesRegressor(n_estimators=300, n_jobs=-1, random_state=rs)
            m["AdaBoost"] = AdaBoostRegressor(n_estimators=150, random_state=rs)
            m["ElasticNet"] = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000)
    if strategy.get("extra_models"):
        m.update(_optional_boosters(task, rs))
    if strategy.get("with_mlp") and task != "time_series":                         # the advisor decides whether it is worth trying (see advisor.plan_tabular)
        m["MLP"] = est_mod.MLPClassifierNet(random_state=rs) if task == "classification" else est_mod.MLPRegressorNet(random_state=rs)
    if task == "time_series" and strategy.get("ts_ref"):
        m.update(_ts_families(strategy, rs))
    return m


# hyper-parameter search spaces
def _tree_space():
    return {"n_estimators": {"type": "randint", "low": 100, "high": 300},
            "max_depth": {"type": "choice", "values": [None, 4, 8, 12, 20]},
            "min_samples_leaf": {"type": "randint", "low": 1, "high": 10},
            "max_features": {"type": "choice", "values": ["sqrt", 0.3, 0.6, 1.0]}}


DEFAULT_SPACES: Dict[str, Dict[str, Dict[str, Any]]] = {
    "TrendSeasonal": {"degree": {"type": "choice", "values": [1, 2]}, "alpha": {"type": "loguniform", "low": 1e-4, "high": 10.0}},
    "MLP": {"h1": {"type": "choice", "values": [16, 32, 64, 128]}, "h2": {"type": "choice", "values": [0, 16, 32, 64]},
            "alpha": {"type": "loguniform", "low": 1e-5, "high": 1e-1}, "learning_rate_init": {"type": "loguniform", "low": 3e-4, "high": 1e-2}},
    "LogisticRegression": {"C": {"type": "loguniform", "low": 1e-3, "high": 1e2}},
    "Ridge": {"alpha": {"type": "loguniform", "low": 1e-2, "high": 1e3}},
    "RandomForest": _tree_space(),
    "ExtraTrees": _tree_space(),
    "DecisionTree": {"max_depth": {"type": "randint", "low": 2, "high": 14}, "min_samples_leaf": {"type": "randint", "low": 1, "high": 30}},
    "KNN": {"n_neighbors": {"type": "randint", "low": 3, "high": 50}, "weights": {"type": "choice", "values": ["uniform", "distance"]}},
    "AdaBoost": {"n_estimators": {"type": "randint", "low": 50, "high": 300}, "learning_rate": {"type": "loguniform", "low": 0.01, "high": 1.0}},
    "ElasticNet": {"alpha": {"type": "loguniform", "low": 1e-3, "high": 10.0}, "l1_ratio": {"type": "uniform", "low": 0.05, "high": 0.95}},
    "HistGradientBoosting": {
        "learning_rate": {"type": "loguniform", "low": 0.01, "high": 0.3},
        "max_leaf_nodes": {"type": "randint", "low": 8, "high": 64},
        "min_samples_leaf": {"type": "randint", "low": 5, "high": 50},
        "l2_regularization": {"type": "loguniform", "low": 1e-3, "high": 10.0},
        "max_iter": {"type": "randint", "low": 100, "high": 400}},
}
DEFAULT_SPACES.update({
    "LinearSVC": {"C": {"type": "loguniform", "low": 1e-2, "high": 1e2}},
    "SGD": {"alpha": {"type": "loguniform", "low": 1e-6, "high": 1e-2}},
    "ComplementNB": {"alpha": {"type": "loguniform", "low": 1e-2, "high": 3.0}},
    "SVC": {"C": {"type": "loguniform", "low": 1e-1, "high": 1e2}, "gamma": {"type": "loguniform", "low": 1e-4, "high": 1.0}},
})
PARAM_BOUNDS_BY_MODEL = {"SGD": {"alpha": (1e-7, 1e-1)}, "ComplementNB": {"alpha": (1e-3, 10.0)}, "SVC": {"gamma": (1e-5, 10.0), "C": (1e-2, 1e3)}}
# Hard bounds: whatever an LLM proposes is clipped into these (it can narrow/shift, never explode).
PARAM_BOUNDS = {"C": (1e-4, 1e3), "alpha": (1e-3, 1e4), "n_estimators": (50, 500), "max_depth": (2, 40),
                "min_samples_leaf": (1, 50), "learning_rate": (0.005, 0.5), "max_leaf_nodes": (4, 128),
                "l2_regularization": (1e-4, 50.0), "max_iter": (50, 600),
                "n_neighbors": (1, 60), "l1_ratio": (0.05, 0.95)}
ALLOWED_STR = {"weights": ("uniform", "distance")}


def sanitize_space(model: str, proposed: Optional[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """Validate a (possibly LLM-proposed) search space against whitelist + bounds.
    Returns (space, notes). Unknown models/params are ignored; invalid specs fall back to defaults."""
    base = DEFAULT_SPACES.get(model, {})
    space, notes = {}, []
    proposed = proposed if isinstance(proposed, dict) else {}
    for p, dflt in base.items():
        spec = proposed.get(p)
        lo_b, hi_b = PARAM_BOUNDS_BY_MODEL.get(model, {}).get(p) or PARAM_BOUNDS.get(p, (None, None))
        ok = None
        try:
            if isinstance(spec, dict):
                t = spec.get("type")
                if t in ("loguniform", "randint", "uniform"):
                    lo, hi = float(spec["low"]), float(spec["high"])
                    lo, hi = max(lo, lo_b), min(hi, hi_b)
                    if lo < hi and (t != "loguniform" or lo > 0):
                        ok = {"type": t, "low": int(lo) if t == "randint" else lo,
                              "high": int(hi) if t == "randint" else hi}
                        if t == "randint" and ok["low"] >= ok["high"]:
                            ok = None
                elif t == "choice":
                    vals = []
                    for v in spec.get("values", []):
                        if p == "max_features":
                            if v in ("sqrt", "log2") or (isinstance(v, (int, float)) and 0.1 <= float(v) <= 1.0):
                                vals.append(v)
                        elif isinstance(v, str) and v in ALLOWED_STR.get(p, ()):
                            vals.append(v)
                        elif v is None and p == "max_depth":
                            vals.append(None)
                        elif isinstance(v, (int, float)) and not isinstance(v, bool) and lo_b is not None \
                                and lo_b <= v <= hi_b:
                            vals.append(v)
                    vals = list(dict.fromkeys(vals))
                    if vals:
                        ok = {"type": "choice", "values": vals}
        except (KeyError, TypeError, ValueError):
            ok = None
        if ok is None:
            if spec is not None:
                notes.append(f"{model}.{p}: proposed range invalid, default used")
            ok = dflt
        space[p] = ok
    return space, notes


def to_distributions(space: Dict[str, Dict[str, Any]], prefix: str) -> Dict[str, Any]:
    out = {}
    for p, s in space.items():
        t = s["type"]
        if t == "loguniform":
            out[prefix + p] = sps.loguniform(s["low"], s["high"])
        elif t == "randint":
            out[prefix + p] = sps.randint(s["low"], s["high"] + 1)
        elif t == "uniform":
            out[prefix + p] = sps.uniform(s["low"], s["high"] - s["low"])
        else:
            out[prefix + p] = list(s["values"])
    return out


def describe_space(space: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    d = {}
    for p, s in space.items():
        d[p] = (f"{s['type']}({s['low']:.4g}, {s['high']:.4g})" if s["type"] != "choice"
                else "choice(" + ", ".join(str(v) for v in s["values"]) + ")")
    return d


# CV helpers
def _cv(task: str, y: np.ndarray, n_splits: int = config.CV_FOLDS):
    if task == "time_series":
        # fewer, longer folds: the first training window is ~half the data (tiny early windows
        # make CV scores on trending series meaninglessly pessimistic)
        k = min(3, max(2, n_splits - 1))
        return TimeSeriesSplit(n_splits=k, test_size=max(5, len(y) // (k * 2)))
    if task == "classification":
        k = int(np.bincount(y).min())
        if k >= 2:
            return StratifiedKFold(n_splits=min(n_splits, k), shuffle=True, random_state=config.RANDOM_STATE)
    return KFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)


def lag1_col(X: pd.DataFrame) -> Optional[str]:
    return next((c for c in X.columns if str(c).endswith("_lag1")), None)


def _build(model, X: pd.DataFrame, task: str, strategy: Dict[str, Any]):
    """Pipeline (+ optional target wrapper). Returns (estimator, param_prefix)."""
    model, spec = _unspec(model)
    ts = getattr(spec, "ts", None)
    if ts and ts["kind"] in ("naive", "seasonal_naive", "trend_seasonal"):         # no learning pipeline: these read the lag / position columns directly
        return model, "model__"
    steps: List[Tuple[str, Any]] = []
    Xp = X
    if spec is not None and spec.ideas:                                  # LLM-proposed feature ideas (validated, executed deterministically)
        from .automl_recipes import IdeaTransformer
        idea = IdeaTransformer(spec.ideas)
        Xp = idea.fit(X).transform(X)
        steps.append(("ideas", idea))
    steps.append(("prep", build_preprocessor(Xp, spec)))
    rec = tuple(getattr(spec, "recipe", ()) or ())
    if "interactions" in rec:
        from .automl_recipes import Interactions
        steps.append(("fe_interactions", Interactions(task=task if task == "classification" else "regression", seed=config.RANDOM_STATE)))
    if "select" in rec:
        from .automl_recipes import SelectTop
        steps.append(("fe_select", SelectTop(task=task if task == "classification" else "regression", seed=config.RANDOM_STATE)))
    prefix = "model__"
    if task == "classification" and strategy.get("imbalance") in imbalance.METHODS:
        model = imbalance.ResamplingClassifier(model, method=strategy["imbalance"], random_state=config.RANDOM_STATE)
        prefix = "model__estimator__"                      # hyper-parameters of the wrapped model are reached through the resampler
    pipe = Pipeline(steps + [("model", model)])
    if ts and ts["kind"] == "delta":
        return est_mod.SeasonalDeltaRegressor(regressor=pipe, col=ts["ref"], log=bool(ts["log"])), "regressor__" + prefix
    if task == "time_series" and strategy.get("delta_target") and lag1_col(X):
        return DeltaRegressor(regressor=pipe, col=lag1_col(X)), "regressor__" + prefix
    if task == "regression" and strategy.get("log_target"):
        return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1), "regressor__" + prefix
    return pipe, prefix


def _py(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


# metrics
def evaluate(task: str, est, X: pd.DataFrame, y: np.ndarray, n_classes: int = 0) -> Dict[str, float]:
    pred = est.predict(X)
    if task == "classification":
        out = {
            "accuracy": accuracy_score(y, pred),
            "balanced_accuracy": balanced_accuracy_score(y, pred),
            "f1_macro": f1_score(y, pred, average="macro", zero_division=0),
            "f1_weighted": f1_score(y, pred, average="weighted", zero_division=0),
        }
        try:
            if hasattr(est, "predict_proba") and len(np.unique(y)) == n_classes and n_classes >= 2:
                proba = est.predict_proba(X)
                out["roc_auc"] = (roc_auc_score(y, proba[:, 1]) if n_classes == 2 else
                                  roc_auc_score(y, proba, multi_class="ovr", average="macro",
                                                labels=list(range(n_classes))))
        except Exception:
            pass
    else:
        out = {"r2": r2_score(y, pred), "rmse": float(np.sqrt(mean_squared_error(y, pred))),
               "mae": mean_absolute_error(y, pred)}
    return {k: round(float(v), 4) for k, v in out.items()}


# training
def make_baseline(task: str, splits: Dict[str, Any], cv=None, scoring: Optional[str] = None) -> Dict[str, Any]:
    """The bar every model must beat: majority / mean, or (time series) the better of mean and last-value. Includes held-out metrics (the critic uses them)."""
    X_tr, X_te, y_tr, y_te = splits["X_train"], splits["X_test"], splits["y_train"], splits["y_test"]
    n_classes = len(splits["classes"] or [])
    cv, scoring = cv or _cv(task, y_tr), scoring or _scoring(task, n_classes)
    dummy = (DummyClassifier(strategy="prior") if task == "classification" else DummyRegressor(strategy="mean"))
    base_pipe = Pipeline([("prep", build_preprocessor(X_tr)), ("model", dummy)])
    base_name = "majority/prior" if task == "classification" else "mean"
    if task == "time_series" and lag1_col(X_tr):         # strongest naive forecast wins as the bar to beat
        naive = LastValueRegressor(col=lag1_col(X_tr))
        c_mean = cross_val_score(base_pipe, X_tr, y_tr, cv=cv, scoring=scoring).mean()
        c_last = cross_val_score(naive, X_tr, y_tr, cv=cv, scoring=scoring).mean()
        if c_last > c_mean:
            base_pipe, base_name = naive, "last value (naive forecast)"
        per = int((splits.get("ts_info") or {}).get("insights", {}).get("period") or 0)
        sref = next((c for c in X_tr.columns if per and str(c).endswith(f"_lag{per}")), None)
        if sref:
            seas = est_mod.SeasonalNaive(col=sref)
            if cross_val_score(seas, X_tr, y_tr, cv=cv, scoring=scoring).mean() > max(c_mean, c_last):
                base_pipe, base_name = seas, f"same period last season (seasonal naive, period {per})"
    base_cv = cross_val_score(base_pipe, X_tr, y_tr, cv=cv, scoring=scoring)
    base_pipe.fit(X_tr, y_tr)
    return {"name": base_name, "cv_score": round(float(base_cv.mean()), 4), **evaluate(task, base_pipe, X_te, y_te, n_classes)}


def fit_default(task: str, splits: Dict[str, Any], strategy: Dict[str, Any]) -> Dict[str, Any]:
    """Baseline + default-hyper-parameter candidates, all scored by CV on train (+ test once)."""
    strategy = enrich_strategy(task, splits, {**DEFAULT_STRATEGY, **(strategy or {})})
    X_tr, X_te, y_tr, y_te = splits["X_train"], splits["X_test"], splits["y_train"], splits["y_test"]
    n_classes = len(splits["classes"] or [])
    cv, scoring = _cv(task, y_tr), _scoring(task, n_classes)

    baseline = make_baseline(task, splits, cv, scoring)

    results: List[Dict[str, Any]] = []
    fitted: Dict[str, Any] = {}
    failed: List[Dict[str, str]] = []
    for name, model in _candidates(task, strategy).items():
        t0 = time.time()
        try:
            est, _ = _build(model, X_tr, task, strategy)
            scores = cross_val_score(clone(est), X_tr, y_tr, cv=cv, scoring=scoring, n_jobs=1, error_score="raise")
            if not np.isfinite(scores).all():
                raise ValueError("cross-validation produced non-finite scores")
            est.fit(X_tr, y_tr)
        except Exception as e:                       # one broken candidate must never poison the ranking: report it and move on
            failed.append({"name": name, "error": f"{type(e).__name__}: {str(e)[:160]}"})
            continue
        fitted[name] = est
        cv_mean = float(scores.mean())
        results.append({"name": name, "cv_mean": round(cv_mean, 4), "cv_std": round(float(scores.std()), 4),
                        "default_cv": round(cv_mean, 4), "select_score": round(cv_mean, 4),
                        "select_source": "default CV", "tuned": False, "tuning": None,
                        "fit_seconds": round(time.time() - t0, 2),
                        "cv_scores": [round(float(v), 4) for v in scores], **evaluate(task, est, X_te, y_te, n_classes)})
    if not results:
        raise RuntimeError("Every candidate model failed: " + "; ".join(f"{f['name']} ({f['error']})" for f in failed))
    return {"baseline": baseline, "candidates": results, "models": fitted, "best": pick_best(results), "failed": failed}


def pick_best(results: List[Dict[str, Any]]) -> str:
    return max(results, key=lambda r: r["select_score"])["name"]       # CV score, never the test set


def tune_candidates(task: str, splits: Dict[str, Any], strategy: Dict[str, Any], results: List[Dict[str, Any]],
                    fitted: Dict[str, Any], spaces: Optional[Dict[str, Dict[str, Any]]] = None,
                    n_iter: Optional[int] = None, top_k: int = config.TUNE_TOP_K) -> Dict[str, Any]:
    """Randomized search + nested CV on the best `top_k` default models. Mutates results/fitted."""
    strategy = {**DEFAULT_STRATEGY, **(strategy or {})}
    n_iter = int(n_iter or strategy.get("tune_n_iter") or config.TUNE_N_ITER)
    X_tr, X_te, y_tr, y_te = splits["X_train"], splits["X_test"], splits["y_train"], splits["y_test"]
    n_classes = len(splits["classes"] or [])
    scoring = _scoring(task, n_classes)
    inner = _cv(task, y_tr, config.TUNE_INNER_CV)
    # time series: outer folds must match the default-model CV exactly, otherwise the comparison is unfair
    outer = _cv(task, y_tr, config.CV_FOLDS if task == "time_series" else config.TUNE_OUTER_CV)
    fresh = _candidates(task, strategy)
    ranked = sorted([r for r in results if space_key(r["name"]) in DEFAULT_SPACES and r["name"] in fresh],
                    key=lambda r: -r["cv_mean"])[:top_k]
    report: Dict[str, Any] = {"enabled": True, "n_iter": n_iter, "inner_cv": inner.get_n_splits(),
                              "outer_cv": outer.get_n_splits(), "scoring": scoring, "models": {}, "notes": []}
    for r in ranked:
        name = r["name"]
        t0 = time.time()
        space, notes = sanitize_space(space_key(name), (spaces or {}).get(name))
        report["notes"] += notes
        est, prefix = _build(fresh[name], X_tr, task, strategy)
        dist = to_distributions(space, prefix)
        search = RandomizedSearchCV(est, dist, n_iter=n_iter, cv=inner, scoring=scoring, refit=True,
                                    random_state=config.RANDOM_STATE, n_jobs=config.N_JOBS, error_score=np.nan)
        nested, err = None, None
        try:
            nested = cross_val_score(clone(search), X_tr, y_tr, cv=outer, scoring=scoring, n_jobs=1)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
        try:
            search.fit(X_tr, y_tr)
        except Exception as e:
            report["models"][name] = {"error": f"{type(e).__name__}: {e}", "used_tuned": False}
            continue
        cvr = search.cv_results_
        trials = []
        for i in np.argsort(cvr["rank_test_score"]):
            trials.append({"rank": int(cvr["rank_test_score"][i]),
                           "params": {k.replace(prefix, ""): _py(v) for k, v in cvr["params"][i].items()},
                           "mean_score": round(float(cvr["mean_test_score"][i]), 4),
                           "std_score": round(float(cvr["std_test_score"][i]), 4),
                           "fit_seconds": round(float(cvr["mean_fit_time"][i]), 3)})
        n_mean = float(np.nanmean(nested)) if nested is not None and not np.all(np.isnan(nested)) else None
        used = n_mean is not None and n_mean > r["default_cv"] + 1e-4
        entry = {"default_cv": r["default_cv"], "inner_best": round(float(search.best_score_), 4),
                 "nested_mean": None if n_mean is None else round(n_mean, 4),
                 "nested_std": None if nested is None else round(float(np.nanstd(nested)), 4),
                 "nested_scores": [] if nested is None else [round(float(v), 4) for v in nested],
                 "best_params": {k.replace(prefix, ""): _py(v) for k, v in search.best_params_.items()},
                 "trials": trials, "space": describe_space(space), "used_tuned": bool(used),
                 "seconds": round(time.time() - t0, 1), "error": err}
        entry["delta_vs_default"] = None if n_mean is None else round(n_mean - r["default_cv"], 4)
        report["models"][name] = entry
        r["tuning"] = entry
        if used:
            fitted[name] = search.best_estimator_
            r.update(tuned=True, select_score=round(n_mean, 4), select_source="nested CV (tuned)",
                     cv_std=entry["nested_std"] if entry["nested_std"] is not None else r["cv_std"],
                     **evaluate(task, search.best_estimator_, X_te, y_te, n_classes))
        else:
            r["select_source"] = "default CV (tuning did not help out-of-fold)"
    return report


def finalize_threshold(task: str, splits: Dict[str, Any], fitted: Dict[str, Any], results: List[Dict[str, Any]], strategy: Dict[str, Any],
                       cost_fn: Optional[float] = None, cost_fp: Optional[float] = None) -> Dict[str, Any]:
    """Binary classification only. Tune the decision threshold of the BEST model (out-of-fold on the training set; see threshold.py) and, when it
    clearly helps, replace that model by a ThresholdedClassifier so every downstream consumer (held-out metrics, leaderboard, saved bundle,
    scoring) uses the tuned threshold. Mutates `fitted` / `results`; returns a JSON-friendly report ({'enabled': False, ...} when not applicable)."""
    strategy = {**DEFAULT_STRATEGY, **(strategy or {})}
    n_classes = len(splits.get("classes") or [])
    if task != "classification" or n_classes != 2 or not strategy.get("tune_threshold"):
        return {"enabled": False, "reason": "binary classification only" if task != "classification" or n_classes != 2 else "disabled"}
    name = pick_best(results)
    row = next(r for r in results if r["name"] == name)
    obj = strategy.get("threshold_objective") or "f1"
    if obj == "cost" and (cost_fn is None or cost_fp is None):
        obj = "f1"
    try:
        out = thr_mod.tune(fitted[name], splits["X_train"], splits["y_train"], splits["X_test"], splits["y_test"], objective=obj,
                           cost_fn=cost_fn, cost_fp=cost_fp, random_state=config.RANDOM_STATE)
    except Exception as e:                                  # threshold tuning must never break a run
        return {"enabled": True, "model": name, "adopted": False, "error": f"{type(e).__name__}: {str(e)[:160]}"}
    rep = {"enabled": True, "model": name, "adopted": out["adopted"], **out["report"]}
    if out["adopted"]:
        before = {k: row.get(k) for k in ("accuracy", "balanced_accuracy", "f1_macro", "f1_weighted")}
        fitted[name] = out["estimator"]
        row.update(evaluate(task, out["estimator"], splits["X_test"], splits["y_test"], n_classes))
        row["threshold"] = out["threshold"]
        rep["metrics_before"] = before
    return rep


# clustering
def cluster_segments(X: pd.DataFrame, k_range=(2, 8)) -> Dict[str, Any]:
    prep = build_preprocessor(X)
    M = prep.fit_transform(X)
    rng = np.random.RandomState(config.RANDOM_STATE)
    sample = rng.choice(len(M), min(len(M), 3000), replace=False)
    rows, best = [], None
    for k in range(k_range[0], min(k_range[1], max(2, len(M) // 20)) + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=config.RANDOM_STATE).fit(M)
        sil = float(silhouette_score(M[sample], km.labels_[sample]))
        dbi = float(davies_bouldin_score(M, km.labels_))
        rows.append({"k": k, "silhouette": round(sil, 4), "davies_bouldin": round(dbi, 4),
                     "inertia": round(float(km.inertia_), 1)})
        if best is None or sil > best[0]:
            best = (sil, k, km)
    sil, k, km = best
    labels = km.labels_
    pipe = Pipeline([("prep", prep), ("model", km)])
    pca = PCA(n_components=2, random_state=config.RANDOM_STATE).fit(M)
    pts = pca.transform(M[sample])
    from . import featurizers as _fz
    _mods = _fz.modalities(X)
    num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat = [c for c in X.columns if c not in num and c not in _mods["text"] and c not in _mods["image"]]   # never put raw text / paths in profiles
    profiles = []
    z = (X[num] - X[num].mean()) / X[num].std().replace(0, 1) if num else pd.DataFrame(index=X.index)
    for c in range(k):
        mask = labels == c
        prof = {"cluster": int(c), "size": int(mask.sum()), "share": round(float(mask.mean()), 4),
                "numeric_means": {col: round(float(X.loc[mask, col].mean()), 3) for col in num[:12]},
                "defining_features": []}
        if num:
            dev = z.loc[mask].mean().sort_values(key=lambda s: -s.abs())
            prof["defining_features"] = [{"feature": f, "z_vs_overall": round(float(v), 2)} for f, v in dev.head(4).items()]
        prof["top_categories"] = {col: str(X.loc[mask, col].mode().iloc[0]) for col in cat[:5] if X.loc[mask, col].notna().any()}
        profiles.append(prof)
    return {"k": int(k), "silhouette": round(sil, 4), "k_scores": rows, "profiles": profiles, "labels": labels,
            "pca_points": pts.tolist(), "pca_labels": labels[sample].tolist(),
            "explained_variance": [round(float(v), 3) for v in pca.explained_variance_ratio_], "pipeline": pipe}


# inference
def predict_with_bundle(bundle: Dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    """Score new raw data with a saved bundle (clustering: cluster id; time_series: needs target history)."""
    pt = bundle.get("problem_type", bundle.get("task"))
    est = bundle["pipeline"]
    if pt == "clustering":
        X = clean_frame(df, bundle["drop_columns"], bundle["datetime_columns"]).reindex(columns=bundle["feature_columns"])
        return pd.DataFrame({"cluster": est.predict(X)}, index=df.index)
    if pt == "time_series":
        plan = {"date_column": bundle["date_column"], "target": bundle["target"], "drop_columns": bundle["drop_columns"],
                **{k: v for k, v in (bundle.get("ts_info") or {}).items() if k in ("lags", "rolls", "freq", "target_agg")}}
        d, _ = prepare_timeseries(df, plan)
        X = clean_frame(d.drop(columns=[bundle["target"]]), bundle["drop_columns"], bundle["datetime_columns"])
        X = X.reindex(columns=bundle["feature_columns"])
        return pd.DataFrame({bundle["date_column"]: d[bundle["date_column"]].to_numpy(), "prediction": est.predict(X)})
    X = df.drop(columns=[c for c in [bundle["target"]] if c in df.columns])
    X = clean_frame(X, bundle["drop_columns"], bundle["datetime_columns"]).reindex(columns=bundle["feature_columns"])
    pred = est.predict(X)
    out = pd.DataFrame(index=df.index)
    if pt == "classification":
        classes = bundle["classes"]
        out["prediction"] = [classes[int(i)] for i in pred]
        if hasattr(est, "predict_proba"):
            proba = est.predict_proba(X)
            for i, c in enumerate(classes[: proba.shape[1]]):
                out[f"proba_{c}"] = proba[:, i].round(4)
    else:
        out["prediction"] = pred
    return out
