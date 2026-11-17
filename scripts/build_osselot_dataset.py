#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2026 Fossology contributors
# SPDX-License-Identifier: LGPL-2.1-only
"""
Build a labelled false-positive dataset from OSSelot curated packages.

Trees are scanned with the copyright agent from before its cleanup stage
(fossology `85b24694d^`). A finding is labelled 0 if it carries a statement the
OSSelot reviewer kept for the file, 1 if it matches nothing kept; partial matches
go to a review file. Packages and trees come from Thesmo's fetch.py.

    python scripts/build_osselot_dataset.py \\
        --agent  /path/to/precleanup/copyright \\
        --conf   /path/to/precleanup/copyright.conf \\
        --thesmo ~/development/community/thesmo --tier quick
"""

import argparse
import csv
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

LOG = logging.getLogger("build_osselot_dataset")

ARCHIVE_SUFFIXES = (".tar", ".tar.gz", ".tgz", ".zip", ".tar.xz", ".tar.bz2")
NO_STATEMENT = (None, "", "NONE", "NOASSERTION")

# Shared by every statement; no evidence for matching.
STOPWORDS = {
    "copyright", "copyrights", "copyrighted", "c", "all", "rights", "right",
    "reserved", "the", "by", "and", "of", "inc", "spdx", "filecopyrighttext",
}

# A finding is real when it carries this much of a curated statement...
REAL_COVERAGE = 0.8
# ...or is mostly made of one that it covers at least half of.
REAL_PARTIAL_OWN = 0.8
REAL_PARTIAL_CURATED = 0.5
# Below this on both measures, it shares nothing meaningful with any.
FP_CEILING = 0.3

# Statement-shaped but unmatched findings go to review, not to training.
STATEMENT_SHAPE = re.compile(
    r"(?i:\bcopyright(?:ed|s)?\b|©|\(c\)|&copy;|&#169;|&#xa9;)"
    r"(?:[\s:,\"'()cC©-]|(?i:&copy;|&#169;|&#xa9;))*"
    # words that introduce a holder ("Copyright Holder: X", "Copyright Notice 2002 X") are read past
    r"(?:(?i:notices?|holders?|owners?|statements?|information|see)[\s:,.-]*)*"
    # and copyright vocabulary, capitalised in headings, never starts a holder itself
    r"(?:(?:19|20)\d\d|(?!(?:See|Notices?|Holders?|Owners?|Information|Status|Act|Treaty|Laws?|Licen[cs]e|"
    r"Statements?|Disclaimers?|Text|Years?|Dates?|All|Rights|Reserved|Office|Policy|Files?|Page|Symbol)\b)"
    r"[A-Z][A-Za-z])"
)


# Template holders (gettext "THE PACKAGE'S COPYRIGHT HOLDER"); decides a label, so
# narrower than SAFAA's PLACEHOLDER_HOLDER_RE.
TEMPLATE_HOLDER = re.compile(
    r"(?:THE PACKAGE'S|\b[\w.+-]+'s) COPYRIGHT HOLDER|M\. Y\. Name|<name of author>"
    r"|[\[<{]\s*copyright holders?\s*[\]>}]"
)
COPYRIGHT_MARKER = re.compile(r"(?i)\bcopyright|©|\(c\)|&copy;|&#169;|&#xa9;")
REAL_EMAIL = re.compile(r"[\w.+-]+@(?!ADDRESS\b)[\w-]+\.[\w.-]+", re.I)


def template_only(text):
    """True when the only holder a real-labelled finding names is a placeholder."""
    return bool(TEMPLATE_HOLDER.search(text)) and not REAL_EMAIL.search(text)


def tokens(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in STOPWORDS}


def judge(finding, curated):
    """(label, score, matched, reason); label None when unsettled. NOASSERTION files are never labelled."""
    if not curated:
        return None, 0.0, "", "no_curated_statement"
    found = tokens(finding)
    best_cur, best_own = 0.0, 0.0
    for statement in curated:
        want = tokens(statement)
        if not want or not found:
            continue
        shared = len(found & want)
        cov_cur, cov_own = shared / len(want), shared / len(found)
        if cov_cur >= REAL_COVERAGE or (cov_own >= REAL_PARTIAL_OWN and cov_cur >= REAL_PARTIAL_CURATED):
            return 0, round(cov_cur, 3), statement, "matches_curated"
        best_cur, best_own = max(best_cur, cov_cur), max(best_own, cov_own)
    if best_cur >= FP_CEILING or best_own >= FP_CEILING:
        return None, round(best_cur, 3), "", "partial_match"
    if STATEMENT_SHAPE.search(finding):
        return None, round(best_cur, 3), "", "statement_shaped_unmatched"
    if not COPYRIGHT_MARKER.search(finding):
        # SPDX-FileCopyrightText values come without the tag.
        return None, round(best_cur, 3), "", "unmarked_unmatched"
    return 1, round(best_cur, 3), "", "unmatched_prose"


def statements(copyright_text):
    if copyright_text in NO_STATEMENT:
        return []
    return [s.strip() for s in copyright_text.splitlines() if s.strip()]


def strip_prefix(files, root):
    """Prefix OSSelot puts before the tree's paths (from Thesmo's benchmark._strip)."""
    parts = files[0]["fileName"].split("/")
    cut = 0
    for i, seg in enumerate(parts):
        if seg.endswith(ARCHIVE_SUFFIXES):
            cut = i + 1
    sample = files[::max(1, len(files) // 400)]
    best, found = cut, -1
    for extra in range(4):
        here = sum(
            1 for f in sample
            if len(f["fileName"].split("/")) > cut + extra
            and os.path.exists(os.path.join(root, *f["fileName"].split("/")[cut + extra:]))
        )
        if here > found:
            best, found = cut + extra, here
    return "/".join(parts[:best]) + "/" if best else ""


def curated_by_file(spdx_path, root):
    """{path relative to root: [curated statement, ...]} for every listed file."""
    with open(spdx_path, encoding="utf-8") as fh:
        files = json.load(fh)["files"]
    prefix = strip_prefix(files, root)
    out = {}
    for f in files:
        name = f["fileName"]
        at = name.find(prefix)
        if at < 0:
            continue
        out[name[at + len(prefix):].lstrip("/")] = statements(f.get("copyrightText"))
    return out


def source_root(extracted, tag):
    """Same rule as Thesmo's `fetch.source_root`."""
    here = os.path.join(extracted, tag)
    if not os.path.isdir(here) or not os.listdir(here):
        return None
    entries = sorted(os.listdir(here))
    if len(entries) == 1 and os.path.isdir(os.path.join(here, entries[0])):
        return os.path.join(here, entries[0])
    return here


def scan(agent, conf_dir, root):
    """{path relative to root: [finding, ...]} from the copyright agent."""
    # Run from conf_dir: the agent reads ./copyright.conf first.
    res = subprocess.run(
        [agent, "-J", "-T", "1", "-d", root],
        cwd=conf_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        errors="replace", check=False,
    )
    if res.returncode != 0:
        raise RuntimeError("agent exited %d: %s" % (res.returncode, res.stderr[-500:]))
    out = {}
    for entry in json.loads(res.stdout):
        results = entry.get("results")
        if not isinstance(results, list):
            continue  # "Unable to read file" and the like
        rel = os.path.relpath(entry["file"], root)
        out[rel] = [r["content"] for r in results if r.get("type") == "statement" and r.get("content")]
    return out


def build_package(row, work, agent, conf_dir, cache_dir, fetcher=None, drop_tree=False):
    """Scan and label one package, cached; optionally fetch it and drop the tree after."""
    tag = row["tag"]
    cached = os.path.join(cache_dir, tag + ".json")
    if os.path.exists(cached):
        with open(cached, encoding="utf-8") as fh:
            got = json.load(fh)
        return tag, (got["rows"], Counter(got["stats"])), None
    spdx = os.path.join(work, "osselot", tag + ".spdx.json")
    root = source_root(os.path.join(work, "extracted"), tag)
    fetched_here = False
    if root is None and fetcher is not None:
        root = fetcher(row)
        fetched_here = root is not None
    if root is None or not os.path.exists(spdx):
        return tag, None, "not fetched"
    try:
        truth = curated_by_file(spdx, root)
        found = scan(agent, conf_dir, root)
    finally:
        if fetched_here and drop_tree:
            shutil.rmtree(os.path.join(work, "extracted", tag), ignore_errors=True)
    rows, stats = [], Counter()
    for rel, findings in found.items():
        if rel not in truth:
            stats["findings_unlisted_file"] += len(findings)
            continue
        curated = truth[rel]
        for text in findings:
            label, score, matched, reason = judge(text, curated)
            stats["findings"] += 1
            stats[reason] += 1
            rows.append({"text": text, "label": label, "package": tag, "file": rel, "score": score,
                         "reason": reason, "matched": matched, "curated": " | ".join(curated)[:500]})
    stats["listed_files"] = len(truth)
    stats["scanned_files_matched"] = sum(1 for r in found if r in truth)
    tmp = cached + ".part"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"rows": rows, "stats": stats}, fh)
    os.replace(tmp, cached)  # a package is cached whole or not at all
    return tag, (rows, stats), None


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", required=True, help="copyright agent binary (pre-cleanup build)")
    ap.add_argument("--conf", required=True, help="copyright.conf from the same revision as --agent")
    ap.add_argument("--agent-rev", default="85b24694d^", help="fossology revision the agent was built from")
    ap.add_argument("--thesmo", required=True, help="Thesmo checkout (packages.json and fetched trees)")
    ap.add_argument("--work", default=None, help="fetched trees (default: <thesmo>/thesmo-validation)")
    ap.add_argument("--tier", choices=("quick", "full", "all"), default="quick")
    ap.add_argument("--out", default=None, help="output directory (default: <work>/safaa-dataset)")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--fetch", action="store_true",
                    help="fetch packages not on disk with Thesmo's tools/validation/fetch.py")
    ap.add_argument("--drop-trees", action="store_true",
                    help="remove each tree this run extracted once it is scanned (archives are kept)")
    ap.add_argument("tags", nargs="*", help="only these packages")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    work = args.work or os.path.join(args.thesmo, "thesmo-validation")
    out_dir = args.out or os.path.join(work, "safaa-dataset")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(args.thesmo, "datasets", "benchmark", "packages.json"), encoding="utf-8") as fh:
        packages = json.load(fh)
    if args.tags:
        packages = [p for p in packages if p["tag"] in args.tags]
    elif args.tier != "all":
        packages = [p for p in packages if p["tier"] == args.tier]

    # Keyed by agent, regexes and labelling rules.
    key = hashlib.sha256()
    for path in (args.agent, args.conf, os.path.abspath(__file__)):
        key.update(sha256(path).encode())
    cache_dir = os.path.join(out_dir, "packages-" + key.hexdigest()[:16])
    os.makedirs(cache_dir, exist_ok=True)
    fetcher = None
    if args.fetch:
        os.environ.setdefault("THESMO_WORK", work)
        sys.path.insert(0, os.path.join(args.thesmo, "tools", "validation"))
        import fetch as thesmo_fetch  # noqa: E402
        fetcher = thesmo_fetch.fetch

    conf_dir = tempfile.mkdtemp(prefix="safaa-agent-conf-")
    shutil.copy(args.conf, os.path.join(conf_dir, "copyright.conf"))
    rows, per_package, skipped = [], {}, {}
    try:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            jobs = [pool.submit(build_package, p, work, os.path.abspath(args.agent), conf_dir,
                                cache_dir, fetcher, args.drop_trees) for p in packages]
            for job in jobs:
                try:
                    tag, result, why = job.result()
                except Exception as e:  # one broken package must not lose the rest
                    LOG.error("package failed: %s", e)
                    continue
                if result is None:
                    skipped[tag] = why
                    LOG.warning("%-10s skipped: %s", tag, why)
                    continue
                pkg_rows, stats = result
                rows.extend(pkg_rows)
                per_package[tag] = dict(stats)
                LOG.info("%-10s %5d findings: real %d, fp %d, review %d, no curation %d", tag,
                         stats["findings"], stats["matches_curated"], stats["unmatched_prose"],
                         stats["partial_match"] + stats["statement_shaped_unmatched"],
                         stats["no_curated_statement"])
    finally:
        shutil.rmtree(conf_dir, ignore_errors=True)

    # Re-judge cached findings with the current rules.
    for r in rows:
        if r["label"] == 0 and template_only(r["text"]):
            r["label"], r["reason"] = None, "template_holder_kept_by_curator"
        elif r["reason"] == "unmatched_prose" and not COPYRIGHT_MARKER.search(r["text"]):
            r["label"], r["reason"] = None, "unmarked_unmatched"
        elif r["reason"] == "statement_shaped_unmatched" and not STATEMENT_SHAPE.search(r["text"]):
            # matched nothing kept, and the guard no longer reads it as a statement
            r["label"], r["reason"] = 1, "unmatched_prose"

    # Once per label; text labelled both ways is dropped.
    by_text = defaultdict(lambda: {"labels": Counter(), "packages": set()})
    for r in rows:
        if r["label"] is not None:
            by_text[r["text"]]["labels"][r["label"]] += 1
            by_text[r["text"]]["packages"].add(r["package"])
    dataset, conflicts = [], 0
    for text, info in by_text.items():
        if len(info["labels"]) > 1:
            conflicts += 1
            continue
        (label, count), = info["labels"].items()
        dataset.append({"text": text, "label": label, "count": count,
                        "packages": ";".join(sorted(info["packages"]))})

    def write(name, items, fields):
        with open(os.path.join(out_dir, name), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(items)

    write("findings.csv", rows, ["package", "file", "label", "reason", "score", "text", "matched", "curated"])
    write("review.csv", [r for r in rows if r["reason"] in ("partial_match", "statement_shaped_unmatched")],
          ["package", "file", "reason", "score", "text", "curated"])
    write("dataset.csv", dataset, ["text", "label", "count", "packages"])

    labels = Counter(d["label"] for d in dataset)
    provenance = {
        "built": datetime.now(timezone.utc).isoformat(),
        "agent_rev": args.agent_rev,
        "agent_sha256": sha256(args.agent),
        "conf_sha256": sha256(args.conf),
        "tier": args.tier if not args.tags else "tags",
        "packages": sorted(per_package),
        "skipped": skipped,
        "thresholds": {"real_coverage": REAL_COVERAGE, "real_partial_own": REAL_PARTIAL_OWN,
                       "real_partial_curated": REAL_PARTIAL_CURATED, "fp_ceiling": FP_CEILING},
        "findings": len(rows),
        "reasons": dict(Counter(r["reason"] for r in rows)),
        "unique_texts": {"real": labels[0], "false_positive": labels[1], "conflicting_dropped": conflicts},
        "per_package": per_package,
    }
    with open(os.path.join(out_dir, "provenance.json"), "w", encoding="utf-8") as fh:
        json.dump(provenance, fh, indent=2)
    LOG.info("dataset: %d unique texts (real %d, fp %d), %d conflicting dropped, %d findings unlabelled -> %s",
             len(dataset), labels[0], labels[1], conflicts, sum(r["label"] is None for r in rows), out_dir)
    return 0 if dataset else 1


if __name__ == "__main__":
    sys.exit(main())
