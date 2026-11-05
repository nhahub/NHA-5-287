"""Text handling on realistic, messy and non-English input. Each case was wrong before (see CHANGES.md)."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from nexusml import nlp
from nexusml.evaluation import metrics


def vocab(text, analyzer="word"):
    v = nlp.TextVec(analyzer=analyzer, ngram_range=(1, 1)).fit([text, text])
    return list(v.vec_.get_feature_names_out()), v


class TestTokenisation(unittest.TestCase):
    def test_hindi_words_are_not_cut_at_vowel_signs(self):
        toks, _ = vocab("भारत एक विशाल और सुंदर देश है जहाँ बहुत भाषाएँ बोली जाती हैं")
        self.assertIn("जहाँ", toks); self.assertIn("भाषाएँ", toks); self.assertNotIn("रत", toks)

    def test_vowelled_arabic_equals_plain_arabic(self):
        a, _ = vocab("الْقَاهِرَةُ هِيَ عَاصِمَةُ جُمْهُورِيَّةِ مِصْرَ الْعَرَبِيَّةِ"); b, _ = vocab("القاهرة هي عاصمة جمهورية مصر العربية")
        self.assertEqual(sorted(a), sorted(b)); self.assertEqual(len(a), 6)

    def test_languages_without_spaces_use_character_ngrams(self):
        for t in ("北京是中华人民共和国的首都", "東京は日本の首都であり大きな都市です", "กรุงเทพมหานครเป็นเมืองหลวงของประเทศไทย"):
            toks, v = vocab(t); self.assertEqual(v.analyzer_, "char_wb"); self.assertGreater(len(toks), 10)
        _, v = vocab("The quick brown fox jumps over the lazy dog"); self.assertEqual(v.analyzer_, "word")

    def test_corpus_stats_use_the_same_tokens(self):
        st = nlp.corpus_stats(["भारत एक विशाल देश है"] * 3); self.assertEqual(st["vocab_size"], 5)

    def test_language_labels_for_non_latin_scripts(self):
        self.assertEqual(nlp.detect_language(["東京は日本の首都であり大きな都市です"] * 5)["language"], "zh/ja")
        self.assertEqual(nlp.detect_language(["こんにちは、元気ですか"] * 5)["language"], "ja")
        self.assertEqual(nlp.detect_language(["भारत एक विशाल देश है"] * 5)["language"], "hi/mr/ne")


class TestCleaning(unittest.TestCase):
    def test_html_and_entities_are_removed(self):
        self.assertEqual(nlp.clean_text("<p><b>Hello&nbsp;world</b> a &lt; b <br/> end</p>"), "Hello world a < b end")

    def test_legitimate_angle_brackets_are_kept(self):
        t = "Returns <class 'int'> when a < b and <Figure size 640x480> is shown"
        self.assertEqual(nlp.clean_text(t), t)

    def test_urls_and_emails_become_placeholders_and_missing_is_empty(self):
        self.assertEqual(nlp.clean_text("see https://x.org/a?b=1 or me@x.org"), "see URLLINK or EMAILADDR")
        self.assertEqual(nlp.clean_text(None), ""); self.assertEqual(nlp.clean_text(float("nan")), "")

    def test_arabic_marks_are_dropped_but_letters_kept(self):
        self.assertEqual(nlp.clean_text("مُحَمَّدٌ"), "محمد")


class TestCandidatesAndScorer(unittest.TestCase):
    def _case(self, df, target):
        d = Path(tempfile.mkdtemp()); p = d / "t.csv"; df.to_csv(p, index=False)
        return {"path": str(p), "truth": {"target": target, "problem_type": "classification"}}

    def _values(self, pm):
        return {"candidate_metrics": [{"name": "m", pm: 0.5}], "best_model_name": "m", "primary_metric": pm, "analysis_plan": {"drop_columns": []}}

    def test_text_only_table_has_a_baseline(self):
        r = np.random.RandomState(0); words = {"a": "alpha beta gamma", "b": "delta epsilon zeta", "c": "eta theta iota"}
        y = r.choice(list(words), 150); df = pd.DataFrame({"text": [f"{words[k]} {r.choice(['one','two','three','four'])} item{i}" for i, k in enumerate(y)], "label": y})
        out = metrics.model_quality(self._case(df, "label"), self._values("f1_macro"), 1)
        self.assertEqual(out["metric"], "f1_macro"); self.assertGreater(out["baseline"], 0.8)           # not a crash, and a baseline that can actually read the text

    def test_multiclass_baseline_uses_the_systems_metric_not_a_one_class_auc(self):
        r = np.random.RandomState(1); y = r.choice(["a", "b", "c"], 300); df = pd.DataFrame({"x": r.normal(size=300) + pd.Series(y).map({"a": 0, "b": 3, "c": 6}).values, "label": y})
        out = metrics.model_quality(self._case(df, "label"), self._values("f1_macro"), 1)
        self.assertGreater(out["baseline"], 0.8)


class TestPersonalDataScan(unittest.TestCase):
    def counts(self, text):
        return nlp.pii_scan([text])["counts"]

    def test_real_formats_are_found(self):
        for k, vals in {"phone": ["+1 415-555-0132", "(415) 555-0132", "+20 100 123 4567", "01012345678", "+44 20 7946 0958", "010 1234 5678"],
                        "card": ["4111 1111 1111 1111", "4111111111111111"], "ipv4": ["192.168.1.10"], "us_ssn": ["123-45-6789"], "email": ["a.b@example.org"]}.items():
            for v in vals:
                self.assertEqual(self.counts(f"contact: {v} please")[k], 1, (k, v))

    def test_numbers_dates_and_versions_are_not_personal_data(self):
        for v in ["1 2 3 4 5 6 7 8 9 10", "2000-10-01 23:00:00", "0 5892.48 736.56 1 6997.32", "3.14159265358979", "12345678901", "version 1.2.3.4.5", "10.0.0.300",
                  "array([ 0.+0.38268343j, 0.-0.92387953j])", "order 1234567890123 shipped"]:
            self.assertEqual(self.counts(v), {}, v)


if __name__ == "__main__":
    unittest.main()
