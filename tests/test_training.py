# SPDX-FileCopyrightText: © 2026 Fossology contributors
# SPDX-License-Identifier: LGPL-2.1-only
"""The release gate and the data handling of the training scripts."""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import retrain_false_positive_detector as rt  # noqa: E402
import train_and_release as tr  # noqa: E402


def report(cand_real=5, cand_noise=30, cur_real=39, cur_noise=100, f1=0.97, newly=()):
    m = {"real_rejected": cand_real, "noise_missed": cand_noise, "fp_f1": f1}
    return {"held_out": {"candidate": m, "shipped": {"real_rejected": cur_real, "noise_missed": cur_noise}},
            "regression": {"newly_failed": list(newly)}}


def silent_report(cand=(3, 14, 0.020), cur=(3, 14, 0.021), newly=()):
    r = report(newly=newly)
    r["held_out_silent"] = {"candidate": dict(zip(("real_rejected", "noise_missed", "flagged_share"), cand)),
                            "shipped": dict(zip(("real_rejected", "noise_missed", "flagged_share"), cur))}
    return r


def test_silent_gate_ignores_boundary_flips_inside_the_review_band():
    r = silent_report()
    r["held_out"]["candidate"]["real_rejected"] = 6   # two more, both flagged for review
    r["held_out"]["shipped"]["real_rejected"] = 4
    assert tr.gate(r, 0.9) == (True, [])


@pytest.mark.parametrize("cand,why", [
    ((4, 14, 0.020), "silently rejects more real copyrights"),
    ((3, 15, 0.020), "silently lets through more noise"),
    ((3, 14, 0.032), "flags more for review"),     # cannot pass by widening the band
])
def test_silent_gate_refuses(cand, why):
    passed, reasons = tr.gate(silent_report(cand=cand), 0.9)
    assert not passed and any(why in r for r in reasons)


def test_gate_passes_a_better_model():
    assert tr.gate(report(), 0.9) == (True, [])


@pytest.mark.parametrize("kwargs,why", [
    ({"cand_real": 40}, "rejects more real copyrights"),
    ({"cand_noise": 101}, "lets through more noise"),
    ({"newly": ["Copyright (c) Luca Barbato"]}, "newly fails"),
    ({"f1": 0.5}, "noise F1"),
])
def test_gate_refuses(kwargs, why):
    passed, reasons = tr.gate(report(**kwargs), 0.9)
    assert not passed and any(why in r for r in reasons)


def test_gate_refuses_trading_real_copyrights_for_noise():
    """Catching all the noise does not buy one more rejected real copyright."""
    passed, _ = tr.gate(report(cand_real=40, cand_noise=0), 0.0)
    assert not passed


@pytest.mark.parametrize("columns,requested,expected", [
    (["text", "label"], None, "text"),
    (["copyright", "falsePositive"], None, "copyright"),
    (["stmt", "fp"], "stmt", "stmt"),
])
def test_resolve_column(columns, requested, expected):
    assert tr.resolve_column(columns, requested, tr.TEXT_COL_CANDIDATES, "text") == expected


def test_resolve_column_fails_loudly():
    with pytest.raises(SystemExit):
        tr.resolve_column(["stmt", "fp"], None, tr.TEXT_COL_CANDIDATES, "text")


def test_corrections_apply_and_refuse_stale(tmp_path):
    legacy = pd.DataFrame({"text": ["a", "b", "c"], "label": [0, 1, 0]})
    good = tmp_path / "good.csv"
    pd.DataFrame([{"line": 2, "action": "relabel", "old_label": 0, "new_label": 1, "reason": "", "copyright": "a"},
                  {"line": 3, "action": "exclude", "old_label": 1, "new_label": "", "reason": "", "copyright": "b"}]
                 ).to_csv(good, index=False)
    fixed, counts = rt.apply_corrections(legacy.copy(), str(good))
    assert fixed.set_index("text")["label"].to_dict() == {"a": 1, "c": 0}
    stale = tmp_path / "stale.csv"
    pd.DataFrame([{"line": 4, "action": "relabel", "old_label": 1, "new_label": 0, "reason": "", "copyright": "c"}]
                 ).to_csv(stale, index=False)
    with pytest.raises(SystemExit, match="lines \\[4\\]"):
        rt.apply_corrections(legacy.copy(), str(stale))


def test_pattern_cap_keeps_distinct_holders():
    df = pd.DataFrame({"text": ["copyright.html Generated using tools/cldr x%d{" % i for i in range(30)]
                       + ["Copyright (C) 2019 Holder Number %s" % h for h in "ABCDEFGHIJ"],
                       "label": [1] * 30 + [0] * 10})
    capped, dropped = rt.cap_patterns(df, 20, 0)
    assert dropped == 10 and (capped["label"] == 0).sum() == 10


def test_shift_meets_the_real_recall_target():
    scores = [i / 100 for i in range(-50, 50)] + [0.9, 0.95]
    labels = [0] * 100 + [1, 1]
    shift = rt.shift_for(scores, labels, 0.99)
    rejected = sum(1 for s, y in zip(scores, labels) if y == 0 and s > shift)
    assert rejected <= 1


# The builder's marker rule must match SAFAA's.
@pytest.mark.parametrize("text", [
    "Copyright 2019 Jane Doe", "(c) 2019 Jane Doe", "© 2019 Jane Doe", "copyrighted by Jane", "Copyright &copy; 2019 X",
    "2007 - 2013, Nicolas François", "(C) Copyright IBM", "no marker here",
])
def test_builder_marker_rule_agrees_with_safaa(text):
    import build_osselot_dataset as builder
    import safaa.Safaa as S
    in_safaa = S.COPYRIGHT_MARKER_RE.search(text) or S.HTML_COPYRIGHT_RE.search(text)
    assert bool(builder.COPYRIGHT_MARKER.search(text)) == bool(in_safaa)
