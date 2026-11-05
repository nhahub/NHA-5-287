"""Goal-driven AutoML search engine (the deterministic half; automl_agent.py is the LLM half that decides WHAT to try).

Ideas adopted from published systems (re-implemented here, not copied):
  FLAML            cheap-to-expensive search, learner chosen with probability ~ 1 / (estimated cost for improvement), local search around the best config
  auto-sklearn 2   a static portfolio warm start, successive halving over a data-size budget, CV vs holdout chosen by data size, ensembles built from the search
  AutoGluon        out-of-fold predictions kept for every model so ensembles/stacks are built without touching the test set
  AIDE             a tree of solutions refined by typed operators (here: search_family / try_recipe / try_ideas / class_weight / ...), with a summarised digest

Protocol
  1. SEARCH   every trial is scored by k-fold CV on the TRAINING split (OOF predictions saved). These scores are optimistic once many configs were tried.
  2. CONFIRM  the finalists are re-scored on FRESH folds (different shuffle) after the search: that is the number the leaderboard and the stop rule use.
  3. TEST     the held-out split is scored once, at the very end. Nothing in the search ever reads it.
"""
import copy
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score, r2_score, roc_auc_score

from . import config, modeling, targets
from .modeling import DEFAULT_SPACES, Spec, sanitize_space, space_key

LOWER_BETTER = {"rmse", "mae"}


# goal
@dataclass
class Goal:
    """What the user asked for. `metric=None` means 'best effort': no threshold, just a good model inside the budget."""
    metric: Optional[str] = None
    bound: str = "min"                       # "min": need value >= x   |   "max": need value <= x (rmse, mae)
    value: Optional[float] = None
    secondary: Optional[Dict[str, Any]] = None
    max_seconds: float = float(config.TARGET_MAX_SECONDS)
    max_rounds: int = 5
    spec: Optional[Dict[str, Any]] = None

    @classmethod
    def from_spec(cls, target_spec: Optional[Dict[str, Any]], task: str, max_seconds: Optional[float] = None, max_rounds: Optional[int] = None) -> "Goal":
        ps = targets.parse(target_spec, task) if target_spec else None
        g = cls(max_seconds=float(max_seconds or (ps or {}).get("max_seconds") or config.TARGET_MAX_SECONDS), max_rounds=int(max_rounds or 5), spec=ps)
        if ps:
            g.metric, g.bound, g.value, g.secondary = ps["metric"], ps["bound"], ps["value"], ps.get("secondary")
        return g

    @property
    def active(self) -> bool:
        return bool(self.metric and self.value is not None)

    def describe(self) -> str:
        if not self.active:
            return "best effort (no threshold given)"
        s = f"{self.metric} {'>=' if self.bound == 'min' else '<='} {self.value}"
        if self.secondary:
            s += f" with {self.secondary['metric']} {'>=' if self.secondary['bound'] == 'min' else '<='} {self.secondary['value']}"
        return s

    def gap(self, v: Optional[float]) -> Optional[float]:
        """Positive = still short of the goal (in the metric's own units)."""
        if not self.active or v is None:
            return None
        return (self.value - v) if self.bound == "min" else (v - self.value)


def goal_value(goal: Goal, task: str, y: np.ndarray, S: np.ndarray, thr: Optional[float] = None) -> Dict[str, Any]:
    """Goal metric on scores S (binary: P(class 1); multiclass: probability matrix; regression: predictions). -> targets.evaluate-style dict."""
    if not goal.active:
        return {}
    if task == "classification" and S.ndim == 2 and S.shape[1] > 2:
        m, k = goal.metric, S.shape[1]
        pred = S.argmax(axis=1)
        from sklearn.metrics import accuracy_score, balanced_accuracy_score
        v = {"accuracy": lambda: accuracy_score(y, pred), "balanced_accuracy": lambda: balanced_accuracy_score(y, pred), "f1": lambda: f1_score(y, pred, average="macro", zero_division=0),
             "roc_auc": lambda: roc_auc_score(y, S, multi_class="ovr", average="macro", labels=list(range(k)))}.get(m)
        if v is None:
            return {"metric": m, "value": None, "met": False, "note": f"{m} is not defined for more than two classes"}
        val = float(v())
        ok = val >= goal.value if goal.bound == "min" else val <= goal.value
        return {"metric": m, "value": round(val, 4), "threshold": None, "met": bool(ok)}
    s = S[:, 1] if (task == "classification" and S.ndim == 2) else S
    return targets.evaluate(goal.spec, y, s, thr)


# trials
@dataclass
class Config:
    family: str
    params: Dict[str, Any] = field(default_factory=dict)
    recipe: Tuple[str, ...] = ()
    ideas: Tuple[str, ...] = ()                     # names of validated ideas (the idea objects live on the session)
    flags: Tuple[str, ...] = ()                     # strategy overrides: class_weight=balanced | regularize | log_target

    def key(self) -> str:
        return json.dumps([self.family, sorted((k, str(v)) for k, v in self.params.items()), sorted(self.recipe), sorted(self.ideas), sorted(self.flags)], default=str)

    def label(self) -> str:
        extra = list(self.recipe) + (["ideas"] if self.ideas else []) + [f.split("=")[0] for f in self.flags]
        return self.family + (" + " + "+".join(extra) if extra else "")


@dataclass
class Trial:
    id: int
    round: int
    cfg: Config
    fidelity: float
    scores: List[float] = field(default_factory=list)
    mean: float = float("nan")
    std: float = float("nan")
    train_mean: Optional[float] = None
    seconds: float = 0.0
    status: str = "ok"
    error: str = ""
    oof: Optional[np.ndarray] = None
    note: str = ""

    def brief(self) -> Dict[str, Any]:
        return {"id": self.id, "round": self.round, "family": self.cfg.family, "label": self.cfg.label(), "params": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)) for k, v in self.cfg.params.items()},
                "recipe": list(self.cfg.recipe), "ideas": list(self.cfg.ideas), "flags": list(self.cfg.flags), "fidelity": self.fidelity, "mean": None if self.status != "ok" else round(self.mean, 4),
                "std": None if self.status != "ok" else round(self.std, 4), "train": None if self.train_mean is None else round(self.train_mean, 4), "seconds": round(self.seconds, 2),
                "status": self.status, "error": self.error[:120], "note": self.note}


# session
class Session:
    """One AutoML search over one prepared training split. All randomness is seeded; parallelism only affects WHEN a trial runs, never its result."""

    def __init__(self, task: str, splits: Dict[str, Any], strategy: Dict[str, Any], goal: Goal, seed: int = 42, n_workers: int = 1,
                 time_budget: Optional[float] = None, log: Optional[Callable[[str], None]] = None):
        self.task, self.splits, self.goal, self.seed = task, splits, goal, seed
        self.strategy = modeling.enrich_strategy(task, splits, {**modeling.DEFAULT_STRATEGY, "with_mlp": True, **(strategy or {})})
        self.plan: Optional[Dict[str, Any]] = None                  # the advisor's insight-driven plan (shortlist, exclusions + reasons, audit)
        self.X0, self.y = splits["X_train"], np.asarray(splits["y_train"])
        self.classes = splits.get("classes") or []
        self.k = len(self.classes)
        self.pm = modeling.primary_metric_for(task if task != "time_series" else "regression", self.k)
        self.folds = list(modeling._cv(task, self.y).split(self.X0, self.y))
        self.oof_mask = np.zeros(len(self.y), bool)
        for _, va in self.folds:
            self.oof_mask[va] = True
        self.n_workers = max(1, int(n_workers))
        self.t0 = time.time()
        self.budget = float(time_budget or goal.max_seconds)
        self.rng = np.random.RandomState(seed)
        self.trials: List[Trial] = []
        self.dropped: List[str] = []
        self.ideas: Dict[str, Dict[str, Any]] = {}              # name -> validated idea
        self.hints: Dict[str, Dict[str, Any]] = {}              # family -> LLM-proposed (sanitized) search-space overrides
        self.effects: List[Dict[str, Any]] = []                 # credit assignment: what each recipe / lever did
        self.log = log or (lambda m: None)
        self._lock = threading.Lock()
        self._factories: Dict[str, Any] = {}
        self._seen: set = set()
        self._tid = -1
        self.big = len(self.y) >= 2000                          # auto-sklearn 2 style policy: successive halving only when data is large
        self.round = 0
        self.ens_method, self.ens_size = "greedy", 15
        self.allow_search = bool(self.strategy.get("tune", True))      # the user switched hyper-parameter tuning off

    # ---------------------------------------------------------------- data / models
    @property
    def X(self) -> pd.DataFrame:
        return self.X0.drop(columns=[c for c in self.dropped if c in self.X0.columns]) if self.dropped else self.X0

    def time_left(self) -> float:
        return self.budget - (time.time() - self.t0)

    def families(self) -> List[str]:
        if not hasattr(self, "_families"):
            self._pool = list(modeling._candidates(self.task, self.strategy).keys())
            self._families = list(self._pool)
            if self.strategy.get("advisor", True) and self.strategy.get("pipeline_kind") not in ("text", "image"):
                try:
                    from . import advisor
                    self.plan = advisor.build_plan(self.task, self.splits, self._pool, self.seed)
                    keep = {e["family"] for e in self.plan["shortlist"]}
                    self._families = [f for f in self._pool if f in keep]
                except Exception as e:                           # the advisor must never stop a run: fall back to the full zoo
                    self.plan = None
                    self.log(f"Model advisor skipped ({type(e).__name__}); trying every family.")
        return list(self._families)

    def audit_exclusions(self, max_families: int = 4, min_seconds: float = 8.0) -> List[Dict[str, Any]]:
        """Verify the advisor's exclusions: fit each excluded family once (default settings, same folds). One that clearly beats the incumbent is promoted and searched,
        so a wrong rule costs a little time, never the result."""
        if not self.plan or self.best() is None:
            return []
        res: List[Dict[str, Any]] = []
        promoted: List[str] = []
        for e in self.plan["excluded"][:max_families]:
            f = e["family"]
            if f not in getattr(self, "_pool", []) or self.time_left() < min_seconds:
                continue
            t = self.evaluate(Config(f), 1.0, rnd=self.round)
            self.record([t])
            inc = self.best()
            ok = t.status == "ok"
            win = bool(ok and t.mean > inc.mean + 0.5 * max(inc.std, 0.002))
            res.append({"family": f, "cv": round(t.mean, 4) if ok else None, "incumbent": round(inc.mean, 4), "promoted": win, "reason_excluded": e["why"]})
            if win:
                promoted.append(f)
                self._families.append(f)
        if promoted and self.time_left() > 5:
            self.run_search(3, families=promoted)
        self.plan["audit"] = res
        return res

    def make_estimator(self, cfg: Config):
        base_model, strat = self._factory_for(cfg)
        m, spec = modeling._unspec(copy.deepcopy(base_model))
        if cfg.params:
            m.set_params(**cfg.params)
        ideas = [self.ideas[n] for n in cfg.ideas if n in self.ideas] or None
        sp = Spec(m, text_rep=getattr(spec, "text_rep", None), image_rep=getattr(spec, "image_rep", None), meta=getattr(spec, "meta", True),
                  recipe=tuple(sorted(set(cfg.recipe) | set(getattr(spec, "recipe", ()) or ()))), ideas=ideas, ts=getattr(spec, "ts", None))
        est, _ = modeling._build(sp, self.X, self.task, strat)
        return est

    def _factory_for(self, cfg: Config):
        strat = dict(self.strategy)
        for f in cfg.flags:
            k, _, v = f.partition("=")
            strat[k] = v or True
        key = json.dumps(sorted((k, str(v)) for k, v in strat.items() if isinstance(v, (str, bool, int, type(None)))))
        if key not in self._factories:
            self._factories[key] = modeling._candidates(self.task, strat)
        return self._factories[key][cfg.family], strat

    # ---------------------------------------------------------------- scoring
    def predict(self, m, X) -> np.ndarray:
        if self.task == "classification":
            if hasattr(m, "predict_proba"):
                P = np.asarray(m.predict_proba(X), dtype=float)
                if P.shape[1] == self.k:
                    return P
            pr = np.asarray(m.predict(X)).astype(int)
            P = np.zeros((len(pr), self.k))
            P[np.arange(len(pr)), pr] = 1.0
            return P
        return np.asarray(m.predict(X), dtype=float)

    def metric(self, y, P) -> float:
        if self.task == "classification":
            if self.pm == "roc_auc":
                return float(roc_auc_score(y, P[:, 1])) if self.k == 2 else float(roc_auc_score(y, P, multi_class="ovr", average="macro", labels=list(range(self.k))))
            return float(f1_score(y, P.argmax(axis=1), average="macro", zero_division=0))
        return float(r2_score(y, P))

    def _subsample(self, tr: np.ndarray, fidelity: float, trial_id: int, fold: int) -> np.ndarray:
        if fidelity >= 0.999:
            return tr
        m = max(30, int(len(tr) * fidelity))
        if self.task == "time_series":
            return tr[-m:]                                      # keep the most recent window
        r = np.random.RandomState(self.seed * 7919 + trial_id * 31 + fold)
        if self.task == "classification":
            from sklearn.model_selection import train_test_split
            try:
                return np.sort(train_test_split(tr, train_size=m, stratify=self.y[tr], random_state=r)[0])
            except ValueError:
                pass
        return np.sort(r.choice(tr, min(m, len(tr)), replace=False))

    def evaluate(self, cfg: Config, fidelity: float = 1.0, folds: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None, trial_id: Optional[int] = None, rnd: Optional[int] = None) -> Trial:
        tid = self.next_id() if trial_id is None else trial_id
        t = Trial(tid, self.round if rnd is None else rnd, cfg, fidelity)
        t0 = time.perf_counter()
        try:
            est = self.make_estimator(cfg)
            X, y = self.X, self.y
            folds = folds or self.folds
            oof = np.full((len(y), self.k) if self.task == "classification" else len(y), np.nan)
            tr_scores = []
            for fi, (tr, va) in enumerate(folds):
                sub = self._subsample(tr, fidelity, tid, fi)
                m = clone(est).fit(X.iloc[sub], y[sub])
                P = self.predict(m, X.iloc[va])
                oof[va] = P
                t.scores.append(self.metric(y[va], P))
                if fi < 2:                                       # train score on two folds is enough to see over-fitting
                    sm = sub[: min(len(sub), 2000)]
                    tr_scores.append(self.metric(y[sm], self.predict(m, X.iloc[sm])))
            t.mean, t.std = float(np.mean(t.scores)), float(np.std(t.scores))
            if not np.isfinite(t.scores).all():
                raise ValueError("non-finite fold score")
            t.train_mean = float(np.mean(tr_scores)) if tr_scores else None
            t.oof = oof if fidelity >= 0.999 else None
        except Exception as e:
            t.status, t.error, t.mean = "failed", f"{type(e).__name__}: {str(e)[:200]}", float("nan")
        t.seconds = time.perf_counter() - t0
        return t

    def next_id(self) -> int:
        with self._lock:
            self._tid += 1
            return self._tid

    def _map(self, fn: Callable, items: List[Any]) -> List[Any]:
        """Evaluate independent configs concurrently. Results come back in INPUT order, so the outcome never depends on thread timing."""
        if self.n_workers <= 1 or len(items) <= 1:
            return [fn(i) for i in items]
        with ThreadPoolExecutor(max_workers=min(self.n_workers, len(items))) as ex:
            return list(ex.map(fn, items))

    def record(self, trials: List[Trial]) -> None:
        for t in trials:
            self.trials.append(t)
            if t.status == "ok" and t.fidelity >= 0.999:
                self._seen.add(t.cfg.key())

    # ---------------------------------------------------------------- views
    def full(self) -> List[Trial]:
        return [t for t in self.trials if t.status == "ok" and t.fidelity >= 0.999 and t.oof is not None]

    def best(self) -> Optional[Trial]:
        f = self.full()
        return max(f, key=lambda t: t.mean) if f else None

    def family_best(self) -> Dict[str, Trial]:
        out: Dict[str, Trial] = {}
        for t in self.full():
            if t.cfg.family not in out or t.mean > out[t.cfg.family].mean:
                out[t.cfg.family] = t
        return out

    # ---------------------------------------------------------------- search-space handling
    def space(self, family: str) -> Dict[str, Dict[str, Any]]:
        base = space_key(family)
        return self.hints.get(family) or DEFAULT_SPACES.get(base) or {}

    def set_hint(self, family: str, proposed: Dict[str, Any]) -> List[str]:
        sp, notes = sanitize_space(space_key(family), proposed)
        if sp:
            self.hints[family] = sp
        return notes

    def _sample(self, s: Dict[str, Any]) -> Any:
        t = s["type"]
        if t == "loguniform":
            return float(np.exp(self.rng.uniform(np.log(s["low"]), np.log(s["high"]))))
        if t == "randint":
            return int(self.rng.randint(s["low"], s["high"] + 1))
        if t == "uniform":
            return float(self.rng.uniform(s["low"], s["high"]))
        return s["values"][self.rng.randint(len(s["values"]))]

    def _perturb(self, s: Dict[str, Any], v: Any, scale: float = 0.25) -> Any:
        t = s["type"]
        try:
            if t == "loguniform":
                lo, hi = np.log(s["low"]), np.log(s["high"])
                return float(np.exp(np.clip(np.log(v) + self.rng.normal(0, scale * (hi - lo)), lo, hi)))
            if t == "randint":
                return int(np.clip(round(v + self.rng.normal(0, max(1.0, scale * (s["high"] - s["low"])))), s["low"], s["high"]))
            if t == "uniform":
                return float(np.clip(v + self.rng.normal(0, scale * (s["high"] - s["low"])), s["low"], s["high"]))
        except (TypeError, ValueError):
            pass
        return v if self.rng.rand() < 0.7 else self._sample(s)

    def propose(self, family: str, base: Optional[Config] = None, explore: Optional[bool] = None) -> Optional[Config]:
        """Blended search (FLAML-style): perturb the family's best config most of the time, sample the prior otherwise. Never repeats a config."""
        sp = self.space(family)
        if not sp:
            return None
        fb = self.family_best().get(family)
        anchor = base or (fb.cfg if fb is not None else None)
        n_fam = sum(1 for t in self.full() if t.cfg.family == family)
        for _ in range(8):
            local = anchor is not None and n_fam >= 2 and (explore is False or (explore is None and self.rng.rand() < 0.7))
            params = {p: (self._perturb(s, anchor.params.get(p), 0.25) if (local and p in anchor.params) else self._sample(s)) for p, s in sp.items()}
            cfg = Config(family, params, anchor.recipe if anchor else (), anchor.ideas if anchor else (), anchor.flags if anchor else ())
            if cfg.key() not in self._seen:
                return cfg
        return None

    # ---------------------------------------------------------------- family selection (simplified ECI)
    def eci(self, family: str) -> float:
        ts = [t for t in self.full() if t.cfg.family == family]
        if not ts:
            return 0.05                                         # unexplored: a fair chance, as in FLAML
        cost = float(np.mean([t.seconds for t in ts])) + 1e-3
        means = [t.mean for t in ts]
        best_so_far = np.maximum.accumulate(means)
        since = len(means) - 1 - int(np.argmax(best_so_far == best_so_far[-1]))
        rate = max((best_so_far[-1] - best_so_far[0]) / max(1, len(means) - 1), 1e-4)
        gap = max(0.0, max(t.mean for t in self.full()) - best_so_far[-1])
        return cost * max(1 + since, gap / rate) + cost

    def pick_families(self, n: int, restrict: Optional[List[str]] = None) -> List[str]:
        fams = [f for f in (restrict or self.families()) if self.space(f)]
        if not fams:
            return []
        w = np.array([1.0 / self.eci(f) for f in fams])
        w = w / w.sum()
        idx = self.rng.choice(len(fams), size=min(n, len(fams)), replace=False, p=w)
        return [fams[i] for i in sorted(idx)]

    # ---------------------------------------------------------------- the search moves
    def portfolio(self) -> List[Trial]:
        """Round 0: every family at its default config. Large data goes through successive halving (25% -> 50% -> 100% of the training rows)."""
        alive = [Config(f) for f in self.families()]
        rungs = [0.25, 0.5, 1.0] if self.big else [1.0]
        out: List[Trial] = []
        for fid in rungs:
            if self.time_left() <= 0:
                break
            ts = self._map(lambda c: self.evaluate(c, fid, rnd=0), alive)
            self.record(ts)
            out += ts
            ok = sorted([t for t in ts if t.status == "ok"], key=lambda t: -t.mean)
            if fid < 0.999:
                alive = [t.cfg for t in ok[:max(3, (len(ok) + 1) // 2)]]
                if not alive:
                    break
        return out

    def run_search(self, n_trials: int, families: Optional[List[str]] = None, explore: Optional[bool] = None) -> List[Trial]:
        if not self.allow_search:
            return []
        """Evaluate up to n_trials new configs in batches of n_workers, family chosen by ECI. Large data: a 40% screen first, promote the promising ones."""
        done: List[Trial] = []
        while len(done) < n_trials and self.time_left() > 0:
            fams = self.pick_families(min(self.n_workers, n_trials - len(done)), families)
            cfgs = [c for c in (self.propose(f, explore=explore) for f in fams) if c is not None]
            if not cfgs:
                break
            if self.big:
                screen = self._map(lambda c: self.evaluate(c, 0.4), cfgs)
                self.record(screen)
                fb = self.family_best()
                keep = [t.cfg for t in screen if t.status == "ok" and (t.cfg.family not in fb or t.mean >= fb[t.cfg.family].mean - 0.03)]
                cfgs = keep or cfgs[:1]
            ts = self._map(lambda c: self.evaluate(c), cfgs)
            self.record(ts)
            done += ts
        return done

    def variant_trials(self, recipe: Tuple[str, ...] = (), ideas: Tuple[str, ...] = (), flags: Tuple[str, ...] = (), families: Optional[List[str]] = None, top: int = 3) -> Dict[str, Any]:
        """Re-run the best config of the top families WITH a recipe / ideas / strategy flag and report what changed (credit assignment for the agent)."""
        fb = sorted(self.family_best().values(), key=lambda t: -t.mean)
        base = [t for t in fb if (families is None or t.cfg.family in families)][:top]
        cfgs, ref = [], []
        for t in base:
            c = Config(t.cfg.family, dict(t.cfg.params), tuple(sorted(set(t.cfg.recipe) | set(recipe))), tuple(sorted(set(t.cfg.ideas) | set(ideas))),
                       tuple(sorted(set(t.cfg.flags) | set(flags))))
            if c.key() not in self._seen:
                cfgs.append(c)
                ref.append(t)
        ts = self._map(lambda c: self.evaluate(c), cfgs)
        self.record(ts)
        deltas = [round(t.mean - r.mean, 4) for t, r in zip(ts, ref) if t.status == "ok"]
        eff = {"recipe": list(recipe), "ideas": list(ideas), "flags": list(flags), "families": [r.cfg.family for r in ref], "deltas": deltas,
               "mean_delta": round(float(np.mean(deltas)), 4) if deltas else None, "round": self.round}
        self.effects.append(eff)
        return eff

    def deepen(self, top: int = 4) -> Dict[str, Any]:
        """Noise control: re-score the top configs on a second set of folds and average (stops the search chasing lucky folds)."""
        alt = list(modeling._cv(self.task, self.y).split(self.X, self.y)) if self.task == "time_series" else self._fresh_folds(self.seed + 17)
        top_t = sorted(self.full(), key=lambda t: -t.mean)[:top]
        ts = self._map(lambda t: self.evaluate(t.cfg, 1.0, folds=alt, rnd=self.round), top_t)
        gaps = [round(t0.mean - t1.mean, 4) for t0, t1 in zip(top_t, ts) if t1.status == "ok"]
        eff = {"deepen": True, "optimism": round(float(np.mean(gaps)), 4) if gaps else None, "round": self.round}
        self.effects.append(eff)
        return eff

    def drop_features(self, cols: List[str]) -> Dict[str, Any]:
        cols = [c for c in cols if c in self.X.columns]
        if not cols or len(cols) >= self.X.shape[1]:
            return {"dropped": [], "note": "nothing to drop"}
        before = self.best().mean if self.best() else None
        self.dropped += cols
        for t in self.trials:                                    # old OOF predictions came from models that saw the dropped columns
            if t.oof is not None:
                t.oof, t.note = None, t.note or "retired: used a dropped column"
        self._seen.clear()
        fb = sorted({t.cfg.family: t for t in sorted(self.trials, key=lambda t: -t.mean if t.status == "ok" else 1)}.values(), key=lambda t: -(t.mean if t.status == "ok" else -1))[:4]
        ts = self._map(lambda t: self.evaluate(Config(t.cfg.family, dict(t.cfg.params), t.cfg.recipe, t.cfg.ideas, t.cfg.flags)), fb)
        self.record(ts)
        after = self.best().mean if self.best() else None
        eff = {"dropped": cols, "score_before": None if before is None else round(before, 4), "score_after": None if after is None else round(after, 4), "round": self.round}
        self.effects.append(eff)
        return eff

    # ---------------------------------------------------------------- folds / diagnostics
    def _fresh_folds(self, seed: int):
        if self.task == "time_series":                       # a different rolling-origin geometry than the search folds (4 shorter windows, not the same 3)
            from sklearn.model_selection import TimeSeriesSplit
            return list(TimeSeriesSplit(n_splits=4, test_size=max(5, len(self.y) // 8)).split(self.X, self.y))
        from sklearn.model_selection import KFold, StratifiedKFold
        if self.task == "classification" and np.bincount(self.y).min() >= len(self.folds):
            return list(StratifiedKFold(len(self.folds), shuffle=True, random_state=seed).split(self.X, self.y))
        return list(KFold(len(self.folds), shuffle=True, random_state=seed).split(self.X, self.y))

    def learning_curve(self, cfg: Config, fractions=(0.25, 0.5, 0.75, 1.0)) -> Dict[str, Any]:
        """CV score vs training-set size for the incumbent: is it data-starved (still rising), capacity-limited (flat and low), or over-fitting (big train/CV gap)?"""
        pts = []
        for f in fractions:
            if f >= 0.999 and self.best() is not None and self.best().cfg.key() == cfg.key():
                t = self.best()
            else:
                t = self.evaluate(cfg, f, rnd=self.round)
            if t.status == "ok":
                pts.append({"fraction": f, "cv": round(t.mean, 4), "train": None if t.train_mean is None else round(t.train_mean, 4)})
        slope = None
        if len(pts) >= 3:
            slope = round((pts[-1]["cv"] - pts[len(pts) // 2 - 1]["cv"]) / max(1e-9, pts[-1]["fraction"] - pts[len(pts) // 2 - 1]["fraction"]), 4)
        last = pts[-1] if pts else {}
        gap = None if last.get("train") is None else round(last["train"] - last["cv"], 4)
        verdict = ("still improving with more data: more rows / a model that uses them would help" if slope is not None and slope > 0.04 else
                   "over-fitting: large train-vs-CV gap, so regularise, simplify or ensemble" if gap is not None and gap > 0.12 else
                   "plateaued: more rows will not help; try different features or a different model family" if slope is not None else "not enough points")
        return {"points": pts, "slope_per_unit_data": slope, "train_cv_gap": gap, "verdict": verdict}

    def ceiling(self) -> Dict[str, Any]:
        """A data-only bound on what ANY model can reach: rows with identical features but different outcomes cannot be separated."""
        try:
            X = self.X
            h = pd.util.hash_pandas_object(X.astype(str).fillna("NA"), index=False)
            g = pd.Series(h.values)
            sizes = g.map(g.value_counts())
            dup_share = float((sizes > 1).mean())
            if dup_share < 0.01 or self.task == "time_series":
                return {"available": False, "duplicate_row_share": round(dup_share, 4)}
            if self.task == "classification":
                df = pd.DataFrame({"h": g.values, "y": self.y})
                lb = float(sum(len(s) - s.value_counts().iloc[0] for _, s in df.groupby("h")["y"]) / len(df))
                return {"available": True, "duplicate_row_share": round(dup_share, 4), "min_error": round(lb, 4), "max_accuracy": round(1 - lb, 4)}
            df = pd.DataFrame({"h": g.values, "y": self.y.astype(float)})
            sse = float(((df.y - df.groupby("h")["y"].transform("mean")) ** 2).sum())
            sst = float(((df.y - df.y.mean()) ** 2).sum()) + 1e-12
            return {"available": True, "duplicate_row_share": round(dup_share, 4), "max_r2": round(1 - sse / sst, 4)}
        except Exception as e:
            return {"available": False, "error": type(e).__name__}
