"""Summarize successful and failed original-checkpoint transfers without hiding failures."""

import argparse
import json
import statistics
import re
from pathlib import Path
from .profiles import PROFILES


def analyze(root):
    rows = [json.loads(p.read_text()) for p in root.glob("*/summary.json")]
    failures = [json.loads(p.read_text()) for p in root.glob("*/failure.json")]
    attempts = rows + failures
    expected = {
        (profile, mode, repeat)
        for profile in PROFILES
        for mode in ("whole", "chunked")
        for repeat in (1, 2, 3)
    }
    assert len(attempts) == 36
    assert {(r["profile"], r["mode"], r["repeat"]) for r in attempts} == expected
    assert rows and all(r["valid"] for r in rows)
    assert all(not r["valid"] for r in failures)
    assert len({r["sha256"] for r in rows}) == 1
    drops = {}
    for row in attempts:
        name = f"{row['profile']}-{row['repeat']}-{row['mode']}"
        counter = re.search(r"dropped (\d+)", (root / name / "netem.txt").read_text())
        count = int(counter.group(1)) if counter else 0
        if row["profile"] in ("loss", "burst"):
            assert count > 0, f"No observed packet drops: {name}"
        drops[name] = count
    result = []
    for profile in PROFILES:
        medians, counts = {}, {}
        for mode in ("whole", "chunked"):
            group = [r for r in rows if r["profile"] == profile and r["mode"] == mode]
            counts[mode] = len(group)
            medians[mode] = (
                statistics.median(r["checkpoint_commit_ms"] for r in group)
                if group
                else None
            )
        ratio = (
            medians["chunked"] / medians["whole"]
            if all(n == 3 for n in counts.values())
            else None
        )
        result.append(
            dict(
                profile=profile, **medians, successful=counts, chunked_over_whole=ratio
            )
        )
    report = dict(
        attempts=36,
        successful=len(rows),
        failed=len(failures),
        failures=failures,
        bytes=rows[0]["bytes"],
        sha256=rows[0]["sha256"],
        medians=result,
        dropped_packets=drops,
        scope="Original trained checkpoint transfer only; no saliency competition or GPU playback",
    )
    (root / "REPORT.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    analyze(parser.parse_args().root)
