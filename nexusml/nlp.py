"""Natural-language pipeline. Everything is classical, deterministic and runs offline (no downloads); sentence embeddings are an OPTIONAL extra.

  representations   word TF-IDF, char n-gram TF-IDF, LSA (dense, for tree models), optional sentence embeddings   -> TextVec / text_transformer
  candidates        representation x classifier grid (LogReg, calibrated LinearSVC, ComplementNB, SGD, RF / HGB on LSA)  -> candidate_specs
                    They are ordinary sklearn Pipelines, so tuning, the critic, the leaderboard, MLflow, the saved bundle and the Predict tab all work unchanged.
  corpus analysis   length / vocabulary / script + language mix, class-distinguishing terms (log-odds with an informative Dirichlet prior),
                    topics (NMF, k chosen by coherence), label-issue suspects (confident-learning style), PII scan, entity-like patterns,
                    lexicon sentiment (English heuristic), representative documents, duplicates                    -> analyze
  after the model   most-confident mistakes, per-document word contributions (model-agnostic occlusion), a text stress test   -> misclassified / explain_document / stress_test
  utilities         extractive summary (TextRank), keyphrases, semantic search index (TF-IDF + LSA)

Privacy: raw text and terms never reach the LLM unless the user opted in (allow_terms); evidence strings carry only counts and shares.
"""
import html as _html
import re
import unicodedata
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge, SGDClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVC

from . import config

NLP_MAX_FEATURES = int(__import__("os").environ.get("NEXUSML_NLP_MAX_FEATURES", "8000"))
SPARSE_REPS = {"word", "char", "wordchar"}
_URL = re.compile(r"https?://\S+|www\.\S+")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
# Python's \w does not match combining marks, so words in Hindi, Thai, Bengali... and vowelled Arabic were cut into fragments ('भारत' -> 'रत'). These ranges keep the marks inside the word:
# Hebrew/Arabic points, Brahmic scripts (Devanagari...Sinhala without the danda punctuation), Thai, Lao, Tibetan, Myanmar, Khmer, combining diacritics, ZWNJ/ZWJ.
_WCH = r"[\w\u0300-\u036f\u0591-\u05bd\u05bf\u05c1\u05c2\u05c4\u05c5\u05c7\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06dc\u06df-\u06e8\u06ea-\u06ed\u0900-\u0963\u0966-\u0dff\u0e00-\u0eff\u0f00-\u0fff\u1000-\u109f\u1780-\u17ff\u200c\u200d]"
_TOKEN = re.compile(_WCH + "+", re.U)
_TOKEN_PATTERN = "(?u)" + _WCH + _WCH + "+"                    # TfidfVectorizer: tokens of >= 2 characters, as before, but mark-aware
_AR_MARKS = re.compile("[\u064b-\u065f\u0670\u0640]")            # Arabic short-vowel marks and tatweel are optional orthography: 'الْقَاهِرَةُ' and 'القاهرة' are the same word
_HTML_TAG = re.compile(r"</?(?:html|head|body|title|meta|link|script|style|div|span|p|br|hr|a|b|i|u|em|strong|small|big|font|center|pre|code|blockquote|ul|ol|li|table|thead|tbody|tr|td|th|h[1-6]|img|section|article|nav|footer|header|form|input|button|label)\b[^<>]*>", re.I)
_NO_SPACE = {"CJK", "HIRAGANA", "KATAKANA", "THAI", "LAO", "KHMER", "MYANMAR", "TIBETAN"}   # scripts written without spaces between words
_SENT = re.compile(r"(?<=[.!?؟۔])\s+|\n+")


def is_sparse_rep(rep: str) -> bool:
    return rep in SPARSE_REPS


def clean_text(s: Any) -> str:
    """HTML tags and entities are removed (markup is not language), URLs and e-mail addresses become placeholders (a model should not memorise an address),
    Arabic optional vowel marks are dropped, whitespace is collapsed."""
    if s is None or (isinstance(s, float) and s != s):
        return ""
    t = str(s)
    if "<" in t and _HTML_TAG.search(t):
        t = _html.unescape(_HTML_TAG.sub(" ", t))
    elif "&" in t and re.search(r"&(?:#\d+|#x[0-9a-fA-F]+|[a-zA-Z]{2,8});", t):
        t = _html.unescape(t)
    t = _AR_MARKS.sub("", _EMAIL.sub(" EMAILADDR ", _URL.sub(" URLLINK ", t)))
    return re.sub(r"\s+", " ", t).strip()


def no_space_share(docs: List[str], sample: int = 300) -> float:
    """Share of letters in scripts that have no spaces between words (Chinese, Japanese, Thai, Lao, Khmer, Myanmar, Tibetan)."""
    mix = script_mix(docs, sample)
    return float(sum(v for k, v in mix.items() if k in _NO_SPACE))


# representations
class TextVec(BaseEstimator, TransformerMixin):
    """ONE text column -> features. analyzer 'word' | 'char_wb'; lsa > 0 adds a truncated SVD (dense, L2-normalised) for models that cannot take sparse input."""

    def __init__(self, analyzer: str = "word", ngram_range: Tuple[int, int] = (1, 2), max_features: Optional[int] = None, lsa: int = 0, clean: bool = True):
        self.analyzer, self.ngram_range, self.max_features, self.lsa, self.clean = analyzer, ngram_range, max_features, lsa, clean

    def _docs(self, X) -> List[str]:
        s = pd.Series(np.asarray(X, dtype=object).ravel() if not isinstance(X, pd.Series) else X.values)
        return [clean_text(v) if self.clean else ("" if v is None or v != v else str(v)) for v in s]

    def fit(self, X, y=None):
        docs = self._docs(X)
        n = len(docs)
        self.analyzer_, self.ngram_range_ = self.analyzer, self.ngram_range
        if self.analyzer == "word" and docs and no_space_share(docs) > 0.3:      # a sentence is one 'word' here: use character n-grams
            self.analyzer_, self.ngram_range_ = "char_wb", (1, 3)
        kw = dict(analyzer=self.analyzer_, ngram_range=self.ngram_range_, max_features=self.max_features or NLP_MAX_FEATURES, token_pattern=_TOKEN_PATTERN,
                  sublinear_tf=True, min_df=2 if n > 150 else 1, max_df=0.95 if n >= 50 else 1.0)
        try:
            self.vec_ = TfidfVectorizer(**kw).fit(docs)
        except ValueError:                                      # empty vocabulary after pruning
            self.vec_ = TfidfVectorizer(**{**kw, "min_df": 1, "max_df": 1.0}).fit([d or "empty" for d in docs])
        self.svd_ = None
        if self.lsa:
            M = self.vec_.transform(docs)
            k = min(int(self.lsa), M.shape[1] - 1, M.shape[0] - 1)
            if k >= 2:
                self.svd_ = TruncatedSVD(n_components=k, random_state=config.RANDOM_STATE).fit(M)
                self.norm_ = Normalizer().fit(self.svd_.transform(M))
        return self

    def transform(self, X):
        M = self.vec_.transform(self._docs(X))
        if self.lsa:
            if self.svd_ is None:
                return M.toarray()
            return self.norm_.transform(self.svd_.transform(M))
        return M

    def get_feature_names_out(self, names=None):
        if self.lsa and self.svd_ is not None:
            return np.array([f"topic_{i}" for i in range(self.svd_.n_components)])
        tag = "char" if getattr(self, "analyzer_", self.analyzer) != "word" else "word"
        return np.array([f"{tag}:{t}" for t in self.vec_.get_feature_names_out()])


class EmbedVec(BaseEstimator, TransformerMixin):
    """Optional sentence embeddings (needs `sentence-transformers`; the model is downloaded on first use). Never selected unless the library is present."""
    _MODEL: Dict[str, Any] = {}

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        from sentence_transformers import SentenceTransformer
        if self.model_name not in EmbedVec._MODEL:
            EmbedVec._MODEL[self.model_name] = SentenceTransformer(self.model_name)
        docs = [clean_text(v) for v in pd.Series(np.asarray(X, dtype=object).ravel())]
        return np.asarray(EmbedVec._MODEL[self.model_name].encode(docs, show_progress_bar=False, normalize_embeddings=True))

    def get_feature_names_out(self, names=None):
        return np.array([f"emb_{i}" for i in range(384)])


def embeddings_available() -> bool:
    if __import__("os").environ.get("NEXUSML_TEXT_EMBEDDINGS", "0") != "1":
        return False
    try:
        import sentence_transformers  # noqa: F401
        return True
    except Exception:
        return False


class WordCharVec(BaseEstimator, TransformerMixin):
    """Word 1-2-gram TF-IDF and char_wb 2-5-gram TF-IDF side by side: words carry meaning, characters absorb typos, morphology and unseen words."""

    def __init__(self, clean: bool = True):
        self.clean = clean

    def fit(self, X, y=None):
        self.w_ = TextVec(analyzer="word", ngram_range=(1, 2), clean=self.clean).fit(X)
        self.c_ = TextVec(analyzer="char_wb", ngram_range=(2, 5), clean=self.clean).fit(X)
        return self

    def transform(self, X):
        from scipy import sparse
        return sparse.hstack([self.w_.transform(X), self.c_.transform(X)]).tocsr()

    def get_feature_names_out(self, names=None):
        return np.concatenate([self.w_.get_feature_names_out(), self.c_.get_feature_names_out()])


class NBLogisticRegression(ClassifierMixin, BaseEstimator):
    """NB-SVM (Wang & Manning 2012): scale each feature by the naive-Bayes log-count ratio of the class vs the rest, then fit a linear model.
    One-vs-rest for several classes. A strong, cheap baseline for sentiment / topic classification on TF-IDF features."""

    def __init__(self, C: float = 4.0, alpha: float = 1.0, class_weight=None, max_iter: int = 2000):
        self.C, self.alpha, self.class_weight, self.max_iter = C, alpha, class_weight, max_iter

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        Xb = X.copy().tocsr()
        Xb.data = np.ones_like(Xb.data)                                    # presence/absence for the ratio, as in the paper
        self.r_, self.models_ = [], []
        targets = [self.classes_[1]] if len(self.classes_) == 2 else list(self.classes_)
        for c in targets:
            pos, neg = np.asarray(Xb[y == c].sum(0)).ravel() + self.alpha, np.asarray(Xb[y != c].sum(0)).ravel() + self.alpha
            r = np.log((pos / pos.sum()) / (neg / neg.sum()))
            m = LogisticRegression(C=self.C, max_iter=self.max_iter, class_weight=self.class_weight).fit(X.multiply(r).tocsr(), (y == c).astype(int))
            self.r_.append(r)
            self.models_.append(m)
        return self

    def _scores(self, X):
        return np.column_stack([m.decision_function(X.multiply(r).tocsr()) for r, m in zip(self.r_, self.models_)])

    def decision_function(self, X):
        d = self._scores(X)
        return d.ravel() if d.shape[1] == 1 else d

    def predict_proba(self, X):
        d = self._scores(X)
        if d.shape[1] == 1:
            p = 1.0 / (1.0 + np.exp(-d[:, 0]))
            return np.column_stack([1 - p, p])
        p = 1.0 / (1.0 + np.exp(-d))
        return p / p.sum(1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


def text_transformer(rep: str):
    if rep == "wordchar":
        return WordCharVec()
    if rep == "char":
        return TextVec(analyzer="char_wb", ngram_range=(2, 5))
    if rep == "lsa":
        return TextVec(analyzer="word", ngram_range=(1, 2), lsa=100)
    if rep == "embed":
        return EmbedVec()
    return TextVec(analyzer="word", ngram_range=(1, 2))


REP_LABEL = {"word": "word TF-IDF", "char": "char n-gram TF-IDF", "wordchar": "word+char TF-IDF", "lsa": "LSA", "embed": "embeddings"}


# models
class CalibratedLinearSVC(ClassifierMixin, BaseEstimator):
    """LinearSVC with calibrated probabilities. `C` stays a top-level parameter so the tuner can search it like any other model."""

    def __init__(self, C: float = 0.5, class_weight=None, random_state: int = 42):
        self.C, self.class_weight, self.random_state = C, class_weight, random_state

    def fit(self, X, y):
        from sklearn.calibration import CalibratedClassifierCV
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        base = LinearSVC(C=self.C, class_weight=self.class_weight, random_state=self.random_state, max_iter=5000)
        cv = int(min(3, np.bincount(np.searchsorted(self.classes_, y)).min()))
        self.cal_ = CalibratedClassifierCV(base, cv=cv, method="sigmoid") if cv >= 2 else base
        self.cal_.fit(X, y)
        try:
            ests = [c.estimator for c in self.cal_.calibrated_classifiers_] if hasattr(self.cal_, "calibrated_classifiers_") else [self.cal_]
            self.coef_ = np.mean([e.coef_ for e in ests], axis=0)
        except Exception:
            pass
        return self

    def predict(self, X):
        return self.cal_.predict(X)

    def predict_proba(self, X):
        if hasattr(self.cal_, "predict_proba"):
            return self.cal_.predict_proba(X)
        d = np.asarray(self.cal_.decision_function(X))
        d = np.c_[-d, d] if d.ndim == 1 else d
        e = np.exp(d - d.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)


def candidate_specs(task: str, strategy: Dict[str, Any], rs: int) -> Dict[str, Any]:
    """Representation x model candidates for a text task. Text-native models see ONLY the text; tree models see LSA + the other columns."""
    from .modeling import Spec
    cw, rg = strategy.get("class_weight"), bool(strategy.get("regularize"))
    L = REP_LABEL
    m: Dict[str, Any] = {}
    if task == "classification":
        c = 0.5 if rg else 4.0
        m[f"LogisticRegression · {L['word']}"] = Spec(LogisticRegression(max_iter=3000, class_weight=cw, C=c), text_rep="word", meta=False)
        m[f"LogisticRegression · {L['char']}"] = Spec(LogisticRegression(max_iter=3000, class_weight=cw, C=c), text_rep="char", meta=False)
        m[f"LinearSVC · {L['word']}"] = Spec(CalibratedLinearSVC(C=0.2 if rg else 0.6, class_weight=cw, random_state=rs), text_rep="word", meta=False)
        m[f"LogisticRegression · {L['wordchar']}"] = Spec(LogisticRegression(max_iter=3000, class_weight=cw, C=c), text_rep="wordchar", meta=False)
        m[f"NB-SVM · {L['word']}"] = Spec(NBLogisticRegression(C=0.5 if rg else 4.0, class_weight=cw), text_rep="word", meta=False)
        m[f"ComplementNB · {L['word']}"] = Spec(ComplementNB(alpha=1.0 if rg else 0.3), text_rep="word", meta=False)
        m[f"SGD · {L['word']}"] = Spec(SGDClassifier(loss="modified_huber", alpha=1e-3 if rg else 2e-4, class_weight=cw, random_state=rs, max_iter=60, tol=1e-3), text_rep="word", meta=False)
        m[f"RandomForest · {L['lsa']}"] = Spec(RandomForestClassifier(n_estimators=200, class_weight=cw, n_jobs=-1, random_state=rs,
                                                                      **({"min_samples_leaf": 3} if rg else {})), text_rep="lsa", meta=True)
        m[f"HistGradientBoosting · {L['lsa']}"] = Spec(HistGradientBoostingClassifier(class_weight=cw, random_state=rs, **({"l2_regularization": 1.0, "max_depth": 4} if rg else {})),
                                                       text_rep="lsa", meta=True)
        if strategy.get("extra_models"):
            m[f"ExtraTrees · {L['lsa']}"] = Spec(ExtraTreesClassifier(n_estimators=300, class_weight=cw, n_jobs=-1, random_state=rs), text_rep="lsa", meta=True)
        if embeddings_available():
            m[f"LogisticRegression · {L['embed']}"] = Spec(LogisticRegression(max_iter=3000, class_weight=cw, C=c), text_rep="embed", meta=False)
    else:
        m[f"Ridge · {L['word']}"] = Spec(Ridge(alpha=3.0 if rg else 1.0), text_rep="word", meta=False)
        m[f"Ridge · {L['char']}"] = Spec(Ridge(alpha=3.0 if rg else 1.0), text_rep="char", meta=False)
        m[f"RandomForest · {L['lsa']}"] = Spec(RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=rs, **({"min_samples_leaf": 3} if rg else {})), text_rep="lsa", meta=True)
        m[f"HistGradientBoosting · {L['lsa']}"] = Spec(HistGradientBoostingRegressor(random_state=rs, **({"l2_regularization": 1.0, "max_depth": 4} if rg else {})), text_rep="lsa", meta=True)
    return m


# corpus statistics
_STOP = set("a an the and or but if of to in on at by for with from as is are was were be been being it its this that these those i you he she we they my your his her our their "
            "not no so do does did have has had will would can could should may might there here which who whom what when where why how than then too very just also".split())
_LANG = {"en": "the and of to in is that it for with as was on are this be by".split(), "fr": "le la les de des et en un une du que qui dans pour est pas sur au avec".split(),
         "es": "el la los las de y en un una que es por con para no se del al como".split(), "de": "der die das und ist nicht ein eine zu den von mit sich auf für dem".split(),
         "it": "il lo la gli le di e che in un una per non con sono del della è si".split(), "pt": "o a os as de e que em um uma para não com é se do da no".split()}


def script_of(ch: str) -> Optional[str]:
    try:
        return unicodedata.name(ch).split()[0]
    except ValueError:
        return None


def script_mix(docs: List[str], sample: int = 300) -> Dict[str, float]:
    cnt: Dict[str, int] = {}
    for d in docs[:sample]:
        for ch in d[:400]:
            if ch.isalpha():
                s = script_of(ch)
                if s:
                    cnt[s] = cnt.get(s, 0) + 1
    tot = max(1, sum(cnt.values()))
    return {k: round(v / tot, 3) for k, v in sorted(cnt.items(), key=lambda kv: -kv[1])[:4]}


def detect_language(docs: List[str], sample: int = 300) -> Dict[str, Any]:
    """Approximate language by stop-word hits (Latin script) or by dominant script otherwise. A heuristic: it says 'unknown' rather than guess."""
    mix = script_mix(docs, sample)
    top = next(iter(mix), None)
    if top and top != "LATIN":
        return {"language": {"ARABIC": "ar/fa/ur", "CYRILLIC": "ru/uk/bg", "CJK": "zh/ja", "HIRAGANA": "ja", "KATAKANA": "ja", "HANGUL": "ko", "HEBREW": "he", "GREEK": "el", "DEVANAGARI": "hi/mr/ne",
                                                                                                          "THAI": "th", "BENGALI": "bn", "TAMIL": "ta", "TELUGU": "te", "ARMENIAN": "hy", "GEORGIAN": "ka"}.get(top, top.lower()), "confidence": "script-based", "scripts": mix}
    words = [w for d in docs[:sample] for w in _TOKEN.findall(d.lower())[:200]]
    if len(words) < 50:
        return {"language": "unknown", "confidence": "too little text", "scripts": mix}
    c = pd.Series(words).value_counts()
    scores = {lg: sum(int(c.get(w, 0)) for w in ws) / len(words) for lg, ws in _LANG.items()}
    lg, sc = max(scores.items(), key=lambda kv: kv[1])
    second = sorted(scores.values())[-2]
    return {"language": lg if sc > 0.04 and sc > 1.3 * second else "unknown", "confidence": "stop-word heuristic", "scripts": mix, "scores": {k: round(v, 3) for k, v in scores.items()}}


def corpus_stats(docs: List[str]) -> Dict[str, Any]:
    toks = [_TOKEN.findall(d.lower()) for d in docs]
    n_tok = np.array([len(t) for t in toks])
    allw = pd.Series([w for t in toks for w in t])
    vc = allw.value_counts() if len(allw) else pd.Series(dtype=int)
    return {"n_docs": len(docs), "n_empty": int((n_tok == 0).sum()), "n_tokens": int(n_tok.sum()), "vocab_size": int(len(vc)),
            "hapax_share": round(float((vc == 1).mean()), 3) if len(vc) else 0.0, "tokens_mean": round(float(n_tok.mean()), 1) if len(n_tok) else 0.0,
            "tokens_median": float(np.median(n_tok)) if len(n_tok) else 0.0, "tokens_p95": float(np.percentile(n_tok, 95)) if len(n_tok) else 0.0,
            "short_docs": int((n_tok < 5).sum()), "type_token_ratio": round(len(vc) / max(1, int(n_tok.sum())), 4), "top_words": [w for w in vc.index if w not in _STOP][:15]}


def top_ngrams(docs: List[str], n: int = 2, k: int = 12) -> List[Dict[str, Any]]:
    try:
        cv = CountVectorizer(ngram_range=(n, n), stop_words="english", min_df=2, max_features=20000)
        M = cv.fit_transform(docs)
        s = np.asarray(M.sum(axis=0)).ravel()
        return [{"term": t, "count": int(s[i])} for i, t in sorted(((i, t) for t, i in cv.vocabulary_.items()), key=lambda it: -s[it[0]])[:k]]
    except ValueError:
        return []


def class_terms(docs: List[str], labels: List[str], top: int = 8, min_count: int = 3) -> Dict[str, List[Dict[str, Any]]]:
    """Terms that distinguish each class: log-odds ratio with an informative Dirichlet prior (Monroe, Colaresi & Quinn 2008); z > 1.96 ~ reliable."""
    try:
        cv = CountVectorizer(min_df=min_count, stop_words="english", max_features=30000)
        M = cv.fit_transform(docs).tocsc()
    except ValueError:
        return {}
    vocab = np.array(cv.get_feature_names_out())
    y = np.asarray(labels)
    tot = np.asarray(M.sum(axis=0)).ravel().astype(float)
    a0 = 500.0
    alpha = a0 * tot / tot.sum()
    out: Dict[str, List[Dict[str, Any]]] = {}
    for c in sorted(set(y)):
        mask = y == c
        yc = np.asarray(M[mask].sum(axis=0)).ravel().astype(float)
        yr = tot - yc
        nc, nr = yc.sum(), yr.sum()
        d = np.log((yc + alpha) / (nc + a0 - yc - alpha)) - np.log((yr + alpha) / (nr + a0 - yr - alpha))
        z = d / np.sqrt(1.0 / (yc + alpha) + 1.0 / (yr + alpha))
        idx = np.argsort(-z)[:top]
        out[str(c)] = [{"term": vocab[i], "z": round(float(z[i]), 2), "count": int(yc[i])} for i in idx if z[i] > 0]
    return out


# topics
def _umass(binary: Any, top_idx: List[int]) -> float:
    s, n = 0.0, 0
    for a in range(1, len(top_idx)):
        for b in range(a):
            wi, wj = top_idx[a], top_idx[b]
            dj = binary[:, wj].sum()
            co = binary[:, wi].multiply(binary[:, wj]).sum()
            s += np.log((co + 1.0) / max(dj, 1.0))
            n += 1
    return s / max(n, 1)


def topics(docs: List[str], labels: Optional[List[str]] = None, k_range=(2, 8), seed: int = 42) -> Dict[str, Any]:
    """NMF topics; k is chosen by UMass coherence of each topic's top words (higher = more interpretable)."""
    if len(docs) < 30:
        return {}
    try:
        vec = TfidfVectorizer(min_df=2, max_df=0.6, stop_words="english", max_features=6000, sublinear_tf=True)
        M = vec.fit_transform(docs)
    except ValueError:
        return {}
    if M.shape[1] < 20:
        return {}
    vocab = np.array(vec.get_feature_names_out())
    binary = (M > 0).astype(float).tocsc()
    rows, best = [], None
    for k in range(k_range[0], min(k_range[1], max(2, len(docs) // 10)) + 1):
        nmf = NMF(n_components=k, init="nndsvda", random_state=seed, max_iter=300).fit(M)
        tops = [list(np.argsort(-nmf.components_[t])[:8]) for t in range(k)]
        coh = float(np.mean([_umass(binary, t) for t in tops]))
        rows.append({"k": k, "coherence": round(coh, 3), "error": round(float(nmf.reconstruction_err_), 3)})
        if best is None or coh > best[0] + 1e-9:
            best = (coh, k, nmf, tops)
    coh, k, nmf, tops = best
    W = nmf.transform(M)
    assign = W.argmax(axis=1)
    out_topics = []
    for t in range(k):
        mask = assign == t
        ex = int(np.argmax(W[:, t])) if mask.any() else None
        entry = {"topic": t, "terms": [vocab[i] for i in tops[t]], "share": round(float(mask.mean()), 3), "n_docs": int(mask.sum()), "exemplar_index": ex}
        if labels is not None and mask.any():
            vc = pd.Series(np.asarray(labels)[mask]).value_counts()
            entry["dominant_label"], entry["label_purity"] = str(vc.index[0]), round(float(vc.iloc[0] / mask.sum()), 3)
        out_topics.append(entry)
    purity = None
    if labels is not None:
        from sklearn.metrics import adjusted_rand_score
        purity = round(float(adjusted_rand_score(np.asarray(labels), assign)), 3)
    return {"k": k, "coherence": round(coh, 3), "k_scores": rows, "topics": out_topics, "assignments": assign.tolist(), "ari_vs_labels": purity}


# label issues, PII, entities, sentiment
def label_issues(docs: List[str], labels: List[str], seed: int = 42, top: int = 12) -> Dict[str, Any]:
    """Confident-learning style: out-of-fold probabilities from a quick linear model; a document is suspect if the model is confident in ANOTHER class
    and its probability for the given label is below that class's typical self-confidence."""
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    y = np.asarray(labels)
    vc = pd.Series(y).value_counts()
    if len(docs) < 40 or vc.min() < 5 or len(vc) < 2:
        return {}
    try:
        M = TfidfVectorizer(min_df=2, sublinear_tf=True, max_features=15000).fit_transform([clean_text(d) for d in docs])
        classes = sorted(vc.index)
        yi = np.searchsorted(classes, y)
        P = cross_val_predict(LogisticRegression(max_iter=1000, C=10), M, yi, cv=StratifiedKFold(min(5, int(vc.min())), shuffle=True, random_state=seed), method="predict_proba")
    except Exception:
        return {}
    self_conf = P[np.arange(len(yi)), yi]
    thr = np.array([self_conf[yi == c].mean() for c in range(len(classes))])
    pred = P.argmax(axis=1)
    sus = [i for i in range(len(yi)) if pred[i] != yi[i] and self_conf[i] < thr[yi[i]] and P[i, pred[i]] >= 0.6]
    sus.sort(key=lambda i: -(P[i, pred[i]] - self_conf[i]))
    return {"n_suspects": len(sus), "share": round(len(sus) / len(yi), 4),
            "suspects": [{"index": int(i), "given": str(classes[yi[i]]), "suggested": str(classes[pred[i]]), "p_given": round(float(self_conf[i]), 3), "p_suggested": round(float(P[i, pred[i]]), 3)} for i in sus[:top]]}


def _luhn(num: str) -> bool:
    d = [int(c) for c in num][::-1]
    return (sum(d[0::2]) + sum(sum(divmod(2 * x, 10)) for x in d[1::2])) % 10 == 0


# Structured patterns, not "any long run of digits": the old phone pattern matched timestamps, decimals and number arrays (126 false phones in 2,000 pandas docstrings with no personal data at all).
_PHONE = re.compile(r"(?<![\w.])(?:\+\d{1,3}[\s.-]?(?:\(?\d{1,4}\)?[\s.-]?){2,5}\d"      # +20 100 123 4567, +44 20 7946 0958
                    r"|\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}"                                   # (415) 555-0132, 415.555.0132
                    r"|0\d{2,4}[\s.-]\d{3,4}[\s.-]?\d{3,5}"                                   # national: 010 1234 5678, 020-123-4567
                    r"|01[0-25]\d{8})(?![\w.]*\d)")                                            # Egyptian mobile 01012345678
_OCTET = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
_PII = {"email": _EMAIL, "phone": _PHONE, "ipv4": re.compile(rf"(?<![\w.])(?:{_OCTET}\.){{3}}{_OCTET}(?![\w]|\.\d)"),
        "us_ssn": re.compile(r"(?<![\w-])\d{3}-\d{2}-\d{4}(?![\w-])"), "iban": re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"),
        "card": re.compile(r"(?<![\w.-])(?:(?:\d{4}[ -]){3}\d{1,7}|[3-6]\d{12,18})(?![\w-]|\.\d)")}      # grouped 4-4-4-4, or 13-19 digits starting 3-6 (then Luhn-checked)


def pii_scan(docs: List[str]) -> Dict[str, Any]:
    """Counts only: matches are never stored or shown. Pattern-based, so it finds the obvious, not everything."""
    counts = {k: 0 for k in _PII}
    affected = set()
    for i, d in enumerate(docs):
        for k, rx in _PII.items():
            hits = rx.findall(d)
            if k == "card":
                hits = [h for h in hits if 13 <= len(re.sub(r"\D", "", h)) <= 19 and _luhn(re.sub(r"\D", "", h))]
            if k == "phone":
                hits = [h for h in hits if 9 <= len(re.sub(r"\D", "", h)) <= 15 and not re.fullmatch(r"\+?\d+\.\d+", h.strip())]       # a signed decimal such as +0.38268343 is a number, not a phone
            if hits:
                counts[k] += len(hits)
                affected.add(i)
    return {"counts": {k: v for k, v in counts.items() if v}, "docs_affected": len(affected), "share": round(len(affected) / max(1, len(docs)), 4)}


_ENT = {"url": _URL, "date": re.compile(r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.? \d{1,2}(?:, \d{4})?)\b"),
        "money": re.compile(r"(?:[$€£]\s?\d[\d,.]*|\b\d[\d,.]*\s?(?:USD|EUR|GBP|EGP)\b)"), "percent": re.compile(r"\b\d+(?:\.\d+)?\s?%"),
        "hashtag": re.compile(r"(?<!\w)#\w+"), "mention": re.compile(r"(?<!\w)@\w+")}
_PROPER = re.compile(r"(?<![.!?]\s)(?<!^)\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")


def entities_lite(docs: List[str]) -> Dict[str, Any]:
    """Pattern-based entities (not a trained NER model): urls, dates, money, percentages, hashtags, @mentions, capitalised multi-word names."""
    counts = {k: sum(len(rx.findall(d)) for d in docs) for k, rx in _ENT.items()}
    names = pd.Series([m for d in docs for m in _PROPER.findall(d)])
    return {"counts": {k: v for k, v in counts.items() if v}, "top_names": names.value_counts().head(10).to_dict() if len(names) else {}}


_POS = set("good great excellent amazing awesome fantastic wonderful love loved loves like liked best perfect happy pleased enjoy enjoyed nice beautiful brilliant superb outstanding "
           "recommend recommended fast easy helpful friendly reliable smooth impressive satisfied delightful fine solid worth positive win success better improved comfortable fresh clean "
           "gorgeous favorite favourite incredible lovely terrific thrilled glad thank thanks".split())
_NEG = set("bad terrible awful horrible worst hate hated dislike poor disappointing disappointed broken useless slow difficult hard angry annoying annoyed waste wasted refund "
           "problem problems issue issues fail failed failure failing rude dirty late never cheap boring unhappy sad worse defective damaged missing error errors crash crashed "
           "scam fraud overpriced mediocre painful ugly stuck negative lost complaint terrible garbage".split())
_NEGATE = {"not", "no", "never", "n't", "cannot", "without", "hardly", "barely"}
_INTENS = {"very": 1.4, "really": 1.3, "extremely": 1.6, "so": 1.2, "absolutely": 1.5, "totally": 1.3, "slightly": 0.6, "somewhat": 0.7}


def sentiment_score(text: str) -> float:
    """Lexicon sentiment in [-1, 1] with negation and intensifiers (VADER-like but tiny). An English-only HEURISTIC, shown as such."""
    toks = _TOKEN.findall(str(text).lower().replace("n't", " not"))
    s, n = 0.0, 0
    for i, w in enumerate(toks):
        v = 1.0 if w in _POS else -1.0 if w in _NEG else 0.0
        if not v:
            continue
        window = toks[max(0, i - 3):i]
        if any(t in _NEGATE for t in window):
            v = -0.8 * v
        for t in window:
            v *= _INTENS.get(t, 1.0)
        s += v
        n += 1
    return float(np.tanh(s / 2.0)) if n else 0.0


# summaries, keyphrases, search
def sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT.split(str(text)) if len(s.strip()) > 2]


def extractive_summary(text: str, n: int = 3) -> List[str]:
    """TextRank: sentences are nodes, TF-IDF cosine similarity are edges; the highest PageRank sentences, kept in their original order."""
    sents = sentences(text)
    if len(sents) <= n:
        return sents
    try:
        M = TfidfVectorizer(stop_words="english").fit_transform(sents)
    except ValueError:
        return sents[:n]
    S = (M @ M.T).toarray()
    np.fill_diagonal(S, 0)
    rs = S.sum(axis=1, keepdims=True)
    P = np.divide(S, rs, out=np.zeros_like(S), where=rs > 0)
    r = np.full(len(sents), 1.0 / len(sents))
    for _ in range(50):
        r2 = 0.15 / len(sents) + 0.85 * P.T @ r
        if np.abs(r2 - r).sum() < 1e-8:
            r = r2
            break
        r = r2
    keep = sorted(np.argsort(-r)[:n])
    return [sents[i] for i in keep]


def keyphrases(docs: List[str], idx: int, k: int = 6) -> List[str]:
    try:
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1, sublinear_tf=True, max_features=20000)
        M = vec.fit_transform(docs)
        row = M[idx].toarray().ravel()
        names = vec.get_feature_names_out()
        out: List[str] = []
        for i in np.argsort(-row):
            if row[i] <= 0 or len(out) >= k:
                break
            if not any(names[i] in o or o in names[i] for o in out):
                out.append(names[i])
        return out
    except ValueError:
        return []


class SemanticIndex:
    """Search a collection by meaning-ish similarity: TF-IDF -> LSA -> cosine. `search(query)` and `similar(i)` return (index, score) pairs."""

    def __init__(self, docs: List[str], dims: int = 128):
        self.vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, ngram_range=(1, 2), min_df=1, max_features=40000)
        M = self.vec.fit_transform([clean_text(d) for d in docs])
        k = min(dims, M.shape[1] - 1, M.shape[0] - 1)
        self.svd = TruncatedSVD(n_components=k, random_state=config.RANDOM_STATE).fit(M) if k >= 2 else None
        self.norm = Normalizer()
        self.E = self.norm.fit_transform(self.svd.transform(M)) if self.svd is not None else Normalizer().fit_transform(M.toarray())

    def _embed(self, texts: List[str]):
        M = self.vec.transform([clean_text(t) for t in texts])
        return self.norm.transform(self.svd.transform(M)) if self.svd is not None else Normalizer().fit_transform(M.toarray())

    def search(self, query: str, k: int = 5) -> List[Tuple[int, float]]:
        s = self.E @ self._embed([query])[0]
        return [(int(i), round(float(s[i]), 3)) for i in np.argsort(-s)[:k] if s[i] > 0]

    def similar(self, i: int, k: int = 5) -> List[Tuple[int, float]]:
        s = self.E @ self.E[i]
        return [(int(j), round(float(s[j]), 3)) for j in np.argsort(-s) if j != i][:k]


# after the model
def _frame_with_text(template: pd.DataFrame, text_col: str, texts: List[str]) -> pd.DataFrame:
    base = pd.concat([template.iloc[[0]]] * len(texts), ignore_index=True)
    base[text_col] = texts
    return base


def misclassified(est, X_te: pd.DataFrame, y_te: np.ndarray, classes: List[str], text_col: str, k: int = 8, snippet: int = 220) -> List[Dict[str, Any]]:
    """The held-out documents the model got wrong with the HIGHEST confidence: usually label noise, ambiguous documents or a missing signal."""
    if not hasattr(est, "predict_proba"):
        return []
    P = np.asarray(est.predict_proba(X_te))
    pred = P.argmax(axis=1)
    wrong = np.where(pred != np.asarray(y_te))[0]
    wrong = wrong[np.argsort(-P[wrong, pred[wrong]])][:k]
    texts = X_te[text_col].astype(str).tolist()
    return [{"row": int(X_te.index[i]) if not isinstance(X_te.index[i], (str,)) else str(X_te.index[i]), "true": classes[int(y_te[i])], "predicted": classes[int(pred[i])],
             "confidence": round(float(P[i, pred[i]]), 3), "text": texts[i][:snippet] + ("…" if len(texts[i]) > snippet else "")} for i in wrong]


def explain_document(est, row: pd.DataFrame, text_col: str, classes: Optional[List[str]] = None, max_tokens: int = 60, top: int = 6) -> Dict[str, Any]:
    """Which WORDS drive this prediction? Model-agnostic occlusion: remove each distinct word and measure how much the predicted class probability
    (or the predicted value) changes. Works for any pipeline; positive = the word supports the prediction."""
    text = str(row[text_col].iloc[0])
    toks = list(dict.fromkeys(w for w in _TOKEN.findall(text) if len(w) > 1))[:max_tokens]
    if not toks:
        return {"supporting": [], "opposing": []}
    variants = [text] + [re.sub(rf"\b{re.escape(w)}\b", " ", text) for w in toks]
    X = _frame_with_text(row, text_col, variants)
    if classes is not None and hasattr(est, "predict_proba"):
        P = np.asarray(est.predict_proba(X))
        c = int(P[0].argmax())
        score, label = P[:, c], classes[c]
    else:
        score, label = np.asarray(est.predict(X), dtype=float), None
    delta = score[0] - score[1:]
    order = np.argsort(-delta)
    sup = [{"word": toks[i], "effect": round(float(delta[i]), 4)} for i in order[:top] if delta[i] > 1e-6]
    opp = [{"word": toks[i], "effect": round(float(delta[i]), 4)} for i in order[::-1][:top] if delta[i] < -1e-6]
    return {"prediction": label, "score": round(float(score[0]), 4), "supporting": sup, "opposing": opp,
            "all_effects": {toks[i]: round(float(delta[i]), 4) for i in range(len(toks)) if abs(delta[i]) > 1e-6}}


def highlight_html(text: str, effects: Dict[str, float]) -> str:
    """The document as HTML with each word shaded by its effect on the prediction (green supports it, red argues against it). Text is escaped."""
    import html as _h
    mx = max([abs(v) for v in effects.values()] + [1e-9])
    out = []
    for part in re.split(r"(\W+)", str(text)):
        v = effects.get(part)
        if v is None or not part.strip():
            out.append(_h.escape(part))
            continue
        a = min(0.85, 0.15 + 0.7 * abs(v) / mx)
        col = f"rgba(46,160,67,{a:.2f})" if v > 0 else f"rgba(215,58,73,{a:.2f})"
        out.append(f"<span style='background:{col};border-radius:3px;padding:0 2px' title='{v:+.3f}'>{_h.escape(part)}</span>")
    return "".join(out)


def _typo(w: str, rng) -> str:
    if len(w) < 4:
        return w
    i = int(rng.randint(1, len(w) - 2))
    return w[:i] + w[i + 1] + w[i] + w[i + 2:]


PERTURBATIONS: Dict[str, Callable] = {
    "typos (10% of words)": lambda t, r: " ".join(_typo(w, r) if r.rand() < 0.10 else w for w in t.split()),
    "UPPERCASE": lambda t, r: t.upper(),
    "drop 20% of words": lambda t, r: " ".join(w for w in t.split() if r.rand() > 0.20) or t,
    "shuffle word order": lambda t, r: " ".join(r.permutation(t.split()).tolist()) if t.split() else t,
    "keep first half": lambda t, r: " ".join(t.split()[: max(1, len(t.split()) // 2)]),
    "add punctuation noise": lambda t, r: " ".join(w + (r.choice(["!", ",", "...", " -"]) if r.rand() < 0.2 else "") for w in t.split()),
}


def stress_test(est, X_te: pd.DataFrame, y_te: np.ndarray, text_col: str, task: str, seed: int = 42, max_rows: int = 600) -> Dict[str, Any]:
    """How much does the winning model's held-out score drop when the TEXT is degraded? (accuracy for classification, R2 for regression)."""
    from sklearn.metrics import accuracy_score, r2_score
    rng = np.random.RandomState(seed)
    n = min(len(X_te), max_rows)
    idx = rng.choice(len(X_te), n, replace=False) if len(X_te) > n else np.arange(len(X_te))
    X, y = X_te.iloc[idx].copy(), np.asarray(y_te)[idx]
    score = (lambda yy, pp: accuracy_score(yy, pp)) if task == "classification" else (lambda yy, pp: r2_score(yy, pp))
    base = float(score(y, est.predict(X)))
    rows = []
    for name, fn in PERTURBATIONS.items():
        r = np.random.RandomState(seed + 1)
        Xp = X.copy()
        Xp[text_col] = [fn(str(t), r) for t in Xp[text_col]]
        try:
            s = float(score(y, est.predict(Xp)))
            rows.append({"perturbation": name, "score": round(s, 4), "drop": round(base - s, 4)})
        except Exception as e:
            rows.append({"perturbation": name, "error": f"{type(e).__name__}"})
    ok = [r for r in rows if "drop" in r]
    worst = max(ok, key=lambda r: r["drop"]) if ok else None
    return {"metric": "accuracy" if task == "classification" else "r2", "clean": round(base, 4), "n_rows": int(n), "results": rows,
            "worst": worst, "fragile": bool(worst and worst["drop"] > 0.10)}


# the analysis stage
def analyze(df: pd.DataFrame, text_col: str, label_col: Optional[str] = None, allow_terms: bool = False, seed: int = 42, max_docs: int = 4000):
    """-> (evidence_items [(source, text, data)], figures [(name, title, caption, fig)], report dict).
    Evidence strings contain only counts / shares (and terms only if allow_terms); the report keeps terms and snippets for the report page and the app."""
    import matplotlib.pyplot as plt
    d = df[[c for c in (text_col, label_col) if c]].dropna(subset=[text_col]).copy()
    if len(d) > max_docs:
        d = d.sample(max_docs, random_state=seed)
    docs = [clean_text(t) for t in d[text_col].astype(str)]
    labels = d[label_col].astype(str).tolist() if label_col and label_col in d else None
    items: List[Tuple[str, str, Any]] = []
    figs: List[Tuple[str, str, str, Any]] = []
    rep: Dict[str, Any] = {"text_column": text_col, "n_docs": len(docs)}

    cs = corpus_stats(docs)
    lang = detect_language(docs)
    rep.update(corpus=cs, language=lang)
    items.append(("nlp_corpus", f"Text corpus: {cs['n_docs']:,} documents, {cs['n_tokens']:,} tokens, vocabulary {cs['vocab_size']:,} (type-token ratio {cs['type_token_ratio']}); "
                                f"median {cs['tokens_median']:.0f} tokens per document (95th percentile {cs['tokens_p95']:.0f}); {cs['n_empty']} empty and {cs['short_docs']} very short (<5 tokens) documents. "
                                f"Language: {lang['language']} ({lang['confidence']}).", cs))
    if cs["short_docs"] / max(1, cs["n_docs"]) > 0.2:
        items.append(("nlp_corpus", f"{cs['short_docs'] / cs['n_docs']:.0%} of documents have fewer than 5 tokens; short texts carry little signal for any text model.", {}))
    # ---- length distribution (figure)
    lens = np.array([len(_TOKEN.findall(t)) for t in docs])
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    if labels:
        for c in sorted(set(labels))[:6]:
            ax.hist(np.clip(lens[np.asarray(labels) == c], 0, np.percentile(lens, 99) or 1), bins=30, alpha=0.55, label=str(c))
        ax.legend(fontsize=8, frameon=False)
    else:
        ax.hist(np.clip(lens, 0, np.percentile(lens, 99) or 1), bins=30, color="#2f6fed", alpha=0.8)
    ax.set_xlabel("tokens per document")
    ax.set_ylabel("documents")
    ax.set_title("Document length")
    figs.append(("length", "Document length distribution", "Length differences between classes can act as a shortcut for a model; check the overlap.", fig))
    if labels is not None:
        ls = pd.Series(lens).groupby(np.asarray(labels)).median()
        if len(ls) >= 2 and ls.max() / max(ls.min(), 1) >= 1.8:
            items.append(("nlp_length", f"Median document length differs by class ({ls.max() / max(ls.min(), 1):.1f}x between longest and shortest class): length alone may predict the label.", ls.to_dict()))
    # ---- bigrams
    rep["top_bigrams"] = top_ngrams(docs, 2, 12)
    # ---- class terms
    if labels is not None and len(set(labels)) >= 2:
        ct = class_terms(docs, labels)
        rep["class_terms"] = ct
        reliable = sum(1 for v in ct.values() for t in v if t["z"] > 1.96)
        txt = f"Class-distinguishing terms found for {len(ct)} classes ({reliable} terms with z > 1.96 on the log-odds scale)."
        if allow_terms:
            txt += " " + "; ".join(f"{c}: {', '.join(t['term'] for t in v[:4])}" for c, v in ct.items())
        items.append(("nlp_class_terms", txt, {"reliable_terms": reliable}))
        if ct:
            cls = list(ct)[:6]
            fig, axes = plt.subplots(1, len(cls), figsize=(3.0 * len(cls), 3.4), squeeze=False)
            for ax, c in zip(axes[0], cls):
                tt = ct[c][:6][::-1]
                ax.barh([t["term"] for t in tt], [t["z"] for t in tt], color="#2f6fed")
                ax.set_title(str(c)[:18], fontsize=9)
                ax.tick_params(labelsize=8)
            fig.suptitle("Terms that distinguish each class (log-odds z)", fontsize=10)
            figs.append(("class_terms", "Class-distinguishing terms", "Higher z = more reliably associated with the class than with the rest.", fig))
    # ---- topics
    tp = topics(docs, labels, seed=seed)
    if tp:
        rep["topics"] = {k: v for k, v in tp.items() if k != "assignments"}
        txt = f"{tp['k']} natural topics found (NMF; k chosen by coherence {tp['coherence']})."
        if tp.get("ari_vs_labels") is not None:
            txt += f" Agreement between topics and labels (adjusted Rand): {tp['ari_vs_labels']}."
        if allow_terms:
            txt += " " + "; ".join(f"T{t['topic']}: {', '.join(t['terms'][:4])}" for t in tp["topics"])
        items.append(("nlp_topics", txt, {"k": tp["k"], "coherence": tp["coherence"]}))
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4), gridspec_kw={"width_ratios": [1, 1.5]})
        a1.plot([r["k"] for r in tp["k_scores"]], [r["coherence"] for r in tp["k_scores"]], "o-", color="#2f6fed")
        a1.axvline(tp["k"], color="#e45b5b", ls="--", lw=1)
        a1.set_xlabel("number of topics")
        a1.set_ylabel("coherence (UMass)")
        a1.set_title("Choosing k", fontsize=10)
        a2.barh([f"T{t['topic']}: {', '.join(t['terms'][:3])}" for t in tp["topics"]][::-1], [t["share"] for t in tp["topics"]][::-1], color="#2f6fed")
        a2.set_xlabel("share of documents")
        a2.set_title("Topics", fontsize=10)
        a2.tick_params(labelsize=8)
        fig.tight_layout()
        figs.append(("topics", "Topics in the collection", "Topic words are shown on this page only; they are not sent to the AI unless you opted in.", fig))
    # ---- label issues
    if labels is not None:
        li = label_issues(docs, labels, seed)
        if li:
            for s in li["suspects"]:
                s["text"] = docs[s["index"]][:200]
            rep["label_issues"] = li
            items.append(("nlp_label_issues", f"Up to {li['n_suspects']} document(s) ({li['share']:.1%}) may be mislabelled or genuinely ambiguous: an out-of-fold model prefers another class. "
                                              "The list is ranked: the first entries are the most likely real errors, later ones are more often just hard or ambiguous cases, so the total is an upper bound, not a count of errors. "
                                              "Review the top of the list before trusting the held-out score; a noisy label caps what any model can reach.", {"n": li["n_suspects"], "share": li["share"]}))
    # ---- PII / entities / sentiment
    pii = pii_scan(docs)
    rep["pii"] = pii
    if pii["docs_affected"]:
        items.append(("nlp_pii", f"Possible personal data in {pii['docs_affected']} document(s) ({pii['share']:.1%}): " + ", ".join(f"{k} x{v}" for k, v in pii["counts"].items())
                      + ". Matches are counted, never stored. Remove or mask personal data before sharing the collection or sending any text to an external service.", pii))
    ent = entities_lite(docs)
    rep["entities"] = ent
    if ent["counts"]:
        items.append(("nlp_entities", "Entity-like patterns: " + ", ".join(f"{k} x{v}" for k, v in ent["counts"].items()) + " (pattern-based, not a trained NER model).", ent["counts"]))
    if lang["language"] == "en" and len(docs) >= 20:
        sc = np.array([sentiment_score(t) for t in docs])
        share = {"positive": float((sc > 0.2).mean()), "neutral": float((np.abs(sc) <= 0.2).mean()), "negative": float((sc < -0.2).mean())}
        rep["sentiment"] = {"share": {k: round(v, 3) for k, v in share.items()}, "mean": round(float(sc.mean()), 3)}
        txt = f"Lexicon sentiment (English heuristic): {share['positive']:.0%} positive, {share['neutral']:.0%} neutral, {share['negative']:.0%} negative."
        if labels is not None and len(set(labels)) <= 8:
            by = pd.Series(sc).groupby(np.asarray(labels)).mean()
            rep["sentiment"]["by_class"] = {str(k): round(float(v), 3) for k, v in by.items()}
            if by.max() - by.min() > 0.25:
                txt += f" Sentiment separates the classes (mean {by.min():.2f} to {by.max():.2f}), so sentiment words will carry much of the signal."
        items.append(("nlp_sentiment", txt, rep["sentiment"]))
    # ---- representative documents (nearest to each class / topic centroid)
    try:
        vec = TfidfVectorizer(stop_words="english", min_df=1, max_features=20000, sublinear_tf=True)
        M = vec.fit_transform(docs)
        groups = labels if labels is not None else (tp["assignments"] if tp else None)
        reps = {}
        if groups is not None:
            g = np.asarray(groups)
            for c in sorted(set(g.tolist()), key=str)[:8]:
                idx = np.where(g == c)[0]
                cen = np.asarray(M[idx].mean(axis=0))
                sims = np.asarray(M[idx] @ cen.T).ravel()
                reps[str(c)] = [{"index": int(idx[j]), "text": docs[idx[j]][:240]} for j in np.argsort(-sims)[:2]]
        rep["representative_docs"] = reps
    except Exception:
        pass
    return items, figs, rep
