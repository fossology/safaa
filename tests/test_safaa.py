# SPDX-FileCopyrightText: © 2026 RAJVEER42 <irajveer.bishnoi2310@gmail.com>
#
# SPDX-License-Identifier: LGPL-2.1-only

import numpy as np
import pandas as pd
import pytest

from safaa.Safaa import SafaaAgent


@pytest.fixture(scope="module")
def agent():
    return SafaaAgent()


# ---------------------------------------------------------------------------
# _ensure_list_of_strings
# ---------------------------------------------------------------------------

class TestEnsureListOfStrings:

    def test_list_of_strings_unchanged(self, agent):
        data = ["Copyright 2024 Foo Inc.", "src/lib/tests"]
        result = agent._ensure_list_of_strings(data)
        assert result == ["Copyright 2024 Foo Inc.", "src/lib/tests"]

    def test_tuple_converted_to_list(self, agent):
        # tuples are a common Python iterable — must not raise AttributeError
        result = agent._ensure_list_of_strings(("Copyright 2024 Foo", "bar"))
        assert result == ["Copyright 2024 Foo", "bar"]

    def test_generator_consumed_to_list(self, agent):
        # generators have no .to_list(); this was broken before the fix
        gen = (s for s in ["Copyright 2024 A", "Copyright 2024 B"])
        result = agent._ensure_list_of_strings(gen)
        assert result == ["Copyright 2024 A", "Copyright 2024 B"]

    def test_pandas_series_converted(self, agent):
        series = pd.Series(["Copyright 2024 Foo", "noise string"])
        result = agent._ensure_list_of_strings(series)
        assert result == ["Copyright 2024 Foo", "noise string"]

    def test_numpy_array_converted(self, agent):
        arr = np.array(["Copyright 2024 Foo", "noise"])
        result = agent._ensure_list_of_strings(arr)
        assert result == ["Copyright 2024 Foo", "noise"]

    def test_list_with_integers_coerced_to_strings(self, agent):
        result = agent._ensure_list_of_strings([2024, 42])
        assert result == ["2024", "42"]

    def test_list_with_none_coerced_to_string(self, agent):
        result = agent._ensure_list_of_strings([None, "Copyright 2024 Foo"])
        assert result == ["None", "Copyright 2024 Foo"]

    def test_list_with_mixed_types(self, agent):
        result = agent._ensure_list_of_strings([1, None, "hello", 3.14])
        assert result == ["1", "None", "hello", "3.14"]

    def test_empty_list(self, agent):
        assert agent._ensure_list_of_strings([]) == []

    def test_empty_tuple(self, agent):
        assert agent._ensure_list_of_strings(()) == []

    def test_empty_generator(self, agent):
        assert agent._ensure_list_of_strings(x for x in []) == []

    def test_single_element_list(self, agent):
        assert agent._ensure_list_of_strings(["only one"]) == ["only one"]

    def test_single_element_tuple(self, agent):
        assert agent._ensure_list_of_strings(("only one",)) == ["only one"]

    def test_output_is_always_list(self, agent):
        # Regardless of input type, output must always be a plain list
        for input_data in [("a",), pd.Series(["a"]), np.array(["a"])]:
            result = agent._ensure_list_of_strings(input_data)
            assert type(result) is list

    def test_whitespace_strings_preserved(self, agent):
        result = agent._ensure_list_of_strings(["  ", "\t"])
        assert result == ["  ", "\t"]


# ---------------------------------------------------------------------------
# predict — baseline smoke tests (broader coverage added in subsequent PRs)
# ---------------------------------------------------------------------------

class TestPredict:

    TRUE_POSITIVES = [
        "Copyright 2024 Siemens AG",
        "Copyright (C) 2019 Red Hat, Inc.",
        "Copyright 2020 Google LLC",
    ]
    FALSE_POSITIVES = [
        "src/lib/c/tests/testlibs",
    ]

    def test_known_true_copyright_predicts_t(self, agent):
        for sample in self.TRUE_POSITIVES:
            result = agent.predict([sample])
            assert result == ["t"], f"Expected 't' for {sample!r}, got {result}"

    def test_known_false_positive_predicts_f(self, agent):
        for sample in self.FALSE_POSITIVES:
            result = agent.predict([sample])
            assert result == ["f"], f"Expected 'f' for {sample!r}, got {result}"

    def test_output_length_matches_input(self, agent):
        data = self.TRUE_POSITIVES + self.FALSE_POSITIVES
        result = agent.predict(data)
        assert len(result) == len(data)

    def test_output_values_only_t_or_f(self, agent):
        data = self.TRUE_POSITIVES + self.FALSE_POSITIVES
        result = agent.predict(data)
        assert all(v in ("t", "f") for v in result)

    def test_accepts_tuple_input(self, agent):
        # Enabled by the _ensure_list_of_strings fix in this PR
        result = agent.predict(tuple(self.TRUE_POSITIVES))
        assert len(result) == len(self.TRUE_POSITIVES)

    def test_accepts_generator_input(self, agent):
        # Enabled by the _ensure_list_of_strings fix in this PR
        result = agent.predict(s for s in self.TRUE_POSITIVES)
        assert len(result) == len(self.TRUE_POSITIVES)

    def test_accepts_pandas_series_input(self, agent):
        series = pd.Series(self.TRUE_POSITIVES + self.FALSE_POSITIVES)
        result = agent.predict(series)
        assert len(result) == len(series)


# ---------------------------------------------------------------------------
# declutter — baseline smoke tests
# ---------------------------------------------------------------------------

class TestDeclutter:

    def test_false_positive_becomes_empty_string(self, agent):
        result = agent.declutter(["src/lib/c/tests/testlibs"], ["f"])
        assert result == [""]

    def test_true_positive_returns_nonempty(self, agent):
        result = agent.declutter(["Copyright 2024 Siemens AG"], ["t"])
        assert result[0] != ""

    def test_output_length_matches_input(self, agent):
        data = ["Copyright 2024 Foo", "noise", "Copyright 2020 Bar"]
        preds = ["t", "f", "t"]
        result = agent.declutter(data, preds)
        assert len(result) == len(data)

    def test_all_false_positives_all_empty(self, agent):
        data = ["noise one", "noise two", "noise three"]
        preds = ["f", "f", "f"]
        result = agent.declutter(data, preds)
        assert result == ["", "", ""]

    def test_empty_input(self, agent):
        assert agent.declutter([], []) == []

    def test_prediction_f_always_wins(self, agent):
        # Even a genuine copyright string should be blanked if predicted false
        result = agent.declutter(["Copyright 2024 Siemens AG"], ["f"])
        assert result == [""]
