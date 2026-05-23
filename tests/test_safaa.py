# SPDX-FileCopyrightText: © 2026 RAJVEER42 <irajveer.bishnoi2310@gmail.com>
#
# SPDX-License-Identifier: LGPL-2.1-only

import warnings

import pytest

from safaa.Safaa import SafaaAgent


@pytest.fixture(scope="module")
def agent():
    return SafaaAgent()


# ---------------------------------------------------------------------------
# predict — threshold argument behavior
# ---------------------------------------------------------------------------

class _FakeProbaClassifier:
    """Test double: predict_proba returns a fixed probability for class 1."""

    def __init__(self, prob_class_one):
        self.prob_class_one = prob_class_one

    def predict_proba(self, X):
        n = X.shape[0]
        p1 = self.prob_class_one
        return [[1.0 - p1, p1] for _ in range(n)]

    def predict(self, X):
        # Not used when predict_proba exists; included so introspection
        # sees a normal estimator interface.
        return [1 if self.prob_class_one >= 0.5 else 0 for _ in range(X.shape[0])]


class TestPredictThreshold:

    SAMPLE = "Copyright 2024 Siemens AG"

    def test_no_warning_with_default_threshold_on_hinge_model(self, agent):
        # Shipped model is SGD(loss='hinge') which has no predict_proba.
        # Calling predict() with the default threshold must NOT warn.
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # promote any warning to an error
            agent.predict([self.SAMPLE])

    def test_warning_fires_on_non_default_threshold_without_predict_proba(self, agent):
        with pytest.warns(UserWarning, match="threshold"):
            agent.predict([self.SAMPLE], threshold=0.3)

    def test_warning_message_mentions_predict_proba(self, agent):
        with pytest.warns(UserWarning) as record:
            agent.predict([self.SAMPLE], threshold=0.7)
        assert any("predict_proba" in str(w.message) for w in record)

    def test_warning_fires_even_at_extreme_thresholds(self, agent):
        for t in (0.0, 1.0, 0.99, 0.01):
            with pytest.warns(UserWarning):
                agent.predict([self.SAMPLE], threshold=t)

    def test_no_warning_when_threshold_explicitly_passed_as_default(self, agent):
        # User explicitly passes 0.5 — equivalent to default, must not warn
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            agent.predict([self.SAMPLE], threshold=0.5)

    def test_prediction_still_returns_valid_output_when_warning_fires(self, agent):
        with pytest.warns(UserWarning):
            result = agent.predict([self.SAMPLE], threshold=0.1)
        assert result in (["t"], ["f"])

    def test_threshold_actually_controls_proba_classifier(self, agent, monkeypatch):
        # Swap in a fake classifier whose probability for class 1 is 0.6.
        # threshold=0.5 → 0.6 >= 0.5 → 'f'
        # threshold=0.7 → 0.6 <  0.7 → 't'
        monkeypatch.setattr(
            agent, "false_positive_detector", _FakeProbaClassifier(prob_class_one=0.6)
        )
        assert agent.predict([self.SAMPLE], threshold=0.5) == ["f"]
        assert agent.predict([self.SAMPLE], threshold=0.7) == ["t"]
        assert agent.predict([self.SAMPLE], threshold=0.3) == ["f"]

    def test_no_warning_when_proba_classifier_used_with_custom_threshold(
        self, agent, monkeypatch
    ):
        # When the loaded model DOES support predict_proba, a non-default
        # threshold is honored and no warning should fire.
        monkeypatch.setattr(
            agent, "false_positive_detector", _FakeProbaClassifier(prob_class_one=0.6)
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            agent.predict([self.SAMPLE], threshold=0.8)

    def test_threshold_boundary_inclusive(self, agent, monkeypatch):
        # Threshold check uses >=, so prediction at the exact boundary should be 'f'
        monkeypatch.setattr(
            agent, "false_positive_detector", _FakeProbaClassifier(prob_class_one=0.5)
        )
        assert agent.predict([self.SAMPLE], threshold=0.5) == ["f"]
