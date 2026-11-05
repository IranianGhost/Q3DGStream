"""Container-only network matrix. Changes only this container's loopback qdisc."""

import json
from pathlib import Path
import platform
import subprocess
import sys

from .profiles import PROFILES

root = Path("data/outputs/priority-matrix")
root.mkdir(parents=True, exist_ok=True)
(root / "environment.json").write_text(
    json.dumps(
        dict(python=sys.version, platform=platform.platform(), profiles=PROFILES),
        indent=2,
    )
)
subprocess.run(
    [sys.executable, "-m", "pytest", "research/quic/priority_transport", "-q"],
    check=True,
)
base = [
    sys.executable,
    "-m",
    "research.quic.priority_transport.benchmark",
    "--cert",
    "cert.pem",
    "--key",
    "key.pem",
]
subprocess.run(
    base
    + ["--scenario", "pipeline", "--frames", "2", "--output", str(root / "pipeline")],
    check=True,
)
for profile, opts in PROFILES.items():
    for seed in range(1, 4):
        for policy in ["fifo", "strict", "weighted"]:
            target = root / f"{profile}-{seed}-{policy}"
            target.mkdir()
            subprocess.run(
                ["tc", "qdisc", "del", "dev", "lo", "root"], capture_output=True
            )
            if opts:
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
                        *opts,
                    ],
                    check=True,
                )
            try:
                with (root / f"{target.name}.log").open("w") as log:
                    subprocess.run(
                        base
                        + [
                            "--profile",
                            profile,
                            "--seed",
                            str(seed),
                            "--policy",
                            policy,
                            "--output",
                            str(target),
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=140,
                    )
                print(f"PASS {target.name}", flush=True)
            finally:
                (target / "netem.txt").write_text(
                    subprocess.check_output(
                        ["tc", "-s", "qdisc", "show", "dev", "lo"], text=True
                    )
                )
                subprocess.run(
                    ["tc", "qdisc", "del", "dev", "lo", "root"], capture_output=True
                )

subprocess.run(
    [sys.executable, "-m", "research.quic.priority_transport.pipeline_matrix"],
    check=True,
)
