"""Exercise the actual adapter helper under each network profile."""

from pathlib import Path
import subprocess
import sys
from .profiles import PROFILES

root = Path("data/outputs/priority-matrix")
root.mkdir(parents=True, exist_ok=True)
for profile, opts in PROFILES.items():
    target = root / f"pipeline-{profile}"
    subprocess.run(["tc", "qdisc", "del", "dev", "lo", "root"], capture_output=True)
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
        subprocess.run(
            [
                sys.executable,
                "-m",
                "research.quic.priority_transport.benchmark",
                "--scenario",
                "pipeline",
                "--frames",
                "2",
                "--profile",
                profile,
                "--cert",
                "cert.pem",
                "--key",
                "key.pem",
                "--output",
                str(target),
            ],
            check=True,
            timeout=140,
        )
    finally:
        target.mkdir(parents=True, exist_ok=True)
        (target / "netem.txt").write_text(
            subprocess.check_output(
                ["tc", "-s", "qdisc", "show", "dev", "lo"], text=True
            )
        )
        subprocess.run(["tc", "qdisc", "del", "dev", "lo", "root"], capture_output=True)
