"""Deployable ensemble models. Kept in their OWN module with no dependency on the search engine, because a pickled model references the module that defines its class:
the exported prediction package vendors this small file instead of the whole AutoML engine."""
from typing import List

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin


class _EnsembleBase(BaseEstimator):
    """`members` arrive already fitted on the full training split (the deployed model). `fit(X, y)` genuinely REFITS clones of them on the data it is given, so any utility
    that does clone(model).fit(train).predict(validation) (cross-validation, learning curves, out-of-fold thresholds) measures real generalisation instead of scoring members on
    rows they already saw."""

    def __init__(self, members=None, weights=None, classes=None, member_names=None):
        self.members, self.weights, self.classes, self.member_names = members, weights, classes, member_names

    def fit(self, X, y=None):
        from sklearn.base import clone
        self.fitted_members_ = [clone(m).fit(X, y) for m in self.members]
        return self

    def _ms(self):
        return getattr(self, "fitted_members_", None) or self.members

    @property
    def classes_(self):
        return np.arange(len(self.classes)) if self.classes is not None else None


class EnsembleClassifier(ClassifierMixin, _EnsembleBase):
    def predict_proba(self, X):
        P = None
        for m, w in zip(self._ms(), self.weights):
            p = np.asarray(m.predict_proba(X), dtype=float)
            P = w * p if P is None else P + w * p
        return P

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


class EnsembleRegressor(RegressorMixin, _EnsembleBase):
    def predict(self, X):
        return sum(w * np.asarray(m.predict(X), dtype=float) for m, w in zip(self._ms(), self.weights))


def _stack_features(P_list: List[np.ndarray], classification: bool) -> np.ndarray:
    cols = [(p[:, :-1] if (classification and p.ndim == 2 and p.shape[1] > 1) else p.reshape(len(p), -1)) for p in P_list]
    return np.hstack(cols)


class _StackBase(_EnsembleBase):
    def __init__(self, members=None, weights=None, classes=None, member_names=None, meta=None):
        super().__init__(members, weights, classes, member_names)
        self.meta = meta

    def fit(self, X, y=None):
        """Real stacking: the meta-learner is trained on OUT-OF-FOLD member predictions, then the members are refitted on all the data."""
        from sklearn.base import clone
        from sklearn.model_selection import cross_val_predict
        clf = self.classes is not None
        Z = _stack_features([cross_val_predict(clone(m), X, y, cv=3, method="predict_proba" if clf else "predict") for m in self.members], clf)
        self.fitted_meta_ = clone(self.meta).fit(Z, y)
        self.fitted_members_ = [clone(m).fit(X, y) for m in self.members]
        return self

    def _meta(self):
        return getattr(self, "fitted_meta_", None) or self.meta


class StackClassifier(ClassifierMixin, _StackBase):
    """Level-2 logistic regression on the members' probabilities (trained on OUT-OF-FOLD predictions only)."""

    def predict_proba(self, X):
        return self._meta().predict_proba(_stack_features([np.asarray(m.predict_proba(X)) for m in self._ms()], True))

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


class StackRegressor(RegressorMixin, _StackBase):
    def predict(self, X):
        return self._meta().predict(_stack_features([np.asarray(m.predict(X), dtype=float) for m in self._ms()], False))
