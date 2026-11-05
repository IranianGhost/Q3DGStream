"""Compare whole-file and chunked transfer of original trained NTC weights."""

from pathlib import Path
import subprocess
import sys
import json
from .profiles import PROFILES

root = Path("data/outputs/real-matrix")
root.mkdir(parents=True, exist_ok=True)
failed = False
for profile, options in PROFILES.items():
    for repeat in range(1, 4):
        # Alternate method order between repeats to reduce fixed-order bias.
        for mode in (["whole", "chunked"] if repeat % 2 else ["chunked", "whole"]):
            target = root / f"{profile}-{repeat}-{mode}"
            subprocess.run(
                ["tc", "qdisc", "del", "dev", "lo", "root"], capture_output=True
            )
            if options:
                subprocess.run(
                    [
                        "tc",
                        "qdisc",
                        "add",
                        "dev",
                        "lo",
                        "root",
                        "netem",
                        "limit",
                        "1000",
                        *options,
                    ],
                    check=True,
                )
            try:
                with (root / f"{target.name}.log").open("w") as log:
                    completed = subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "research.quic.priority_transport.real_benchmark",
                            "--scene",
                            "data/scenes/flame_steak",
                            "--cert",
                            "cert.pem",
                            "--key",
                            "key.pem",
                            "--mode",
                            mode,
                            "--profile",
                            profile,
                            "--repeat",
                            str(repeat),
                            "--output",
                            str(target),
                        ],
                        check=False,
                        timeout=260,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                if completed.returncode:
                    failed = True
                    target.mkdir(exist_ok=True)
                    if not (target / "failure.json").exists():
                        (target / "failure.json").write_text(
                            json.dumps(
                                dict(
                                    mode=mode,
                                    profile=profile,
                                    repeat=repeat,
                                    valid=False,
                                    error=f"subprocess exit {completed.returncode}; see log",
                                ),
                                indent=2,
                            )
                        )
                print(
                    "FAIL" if completed.returncode else "PASS", target.name, flush=True
                )
            finally:
                target.mkdir(exist_ok=True)
                (target / "netem.txt").write_text(
                    subprocess.check_output(
                        ["tc", "-s", "qdisc", "show", "dev", "lo"], text=True
                    )
                )
                subprocess.run(
                    ["tc", "qdisc", "del", "dev", "lo", "root"], capture_output=True
                )

if failed:
    raise SystemExit(1)
