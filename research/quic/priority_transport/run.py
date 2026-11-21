"""Run an NTC submission method against Quinn's stream priorities."""

import argparse
from pathlib import Path
import subprocess
import sys


METHODS = {
    "original": "original",
    "sequential": "sequential",
    "concurrent": "concurrent",
    "p0-first": "p0_first",
    "fifo": "amirali_fifo",
    "strict": "amirali_strict",
    "weighted": "amirali_weighted",
}


def main():
    parser = argparse.ArgumentParser(
        description="Compare how application writes interact with Quinn prioritization."
    )
    parser.add_argument("method", choices=[*METHODS, "all"])
    parser.add_argument("--quinn", choices=["equal", "tiered", "both"])
    parser.add_argument("--frames", type=int)
    parser.add_argument("--reps", type=int)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--rate-mbps", type=float, default=30)
    parser.add_argument("--delay-ms", type=float, default=10)
    parser.add_argument("--loss-pct", type=float, default=0)
    parser.add_argument("--tier-delays-ms",
                        help="Sender delays for P0,P1,P2,P3, e.g. 0,250,500,750")
    parser.add_argument("--progressive", action="store_true",
                        help="Play on P0, then show each later group")
    parser.add_argument("--show-window", action="store_true",
                        help="Open the progressive preview in your browser")
    parser.add_argument("--save-stages", action="store_true",
                        help="Save every displayed priority level as a PNG")
    parser.add_argument("--no-netns", action="store_true", help="Use ordinary loopback, without network emulation")
    parser.add_argument("--dry-run", action="store_true")
    args, extra = parser.parse_known_args()

    methods = list(METHODS.values()) if args.method == "all" else [METHODS[args.method]]
    policy = args.quinn or ("tiered" if args.progressive else "both")
    policies = ["equal", "tiered"] if policy == "both" else [policy]
    runner = Path(__file__).with_name("run_scheduling_experiment.py")
    delays = args.tier_delays_ms or ("0,250,500,750" if args.progressive else "0,0,0,0")
    try:
        values = [int(part.strip()) for part in delays.split(",")]
    except ValueError:
        parser.error("--tier-delays-ms must have four integer values")
    if len(values) != 4 or any(not 0 <= value <= 60_000 for value in values):
        parser.error("--tier-delays-ms needs four values between 0 and 60000")
    if args.show_window and not args.progressive:
        parser.error("--show-window requires --progressive")
    if args.save_stages and not args.progressive:
        parser.error("--save-stages requires --progressive")

    command = [
        sys.executable, str(runner),
        "--modes", ",".join(methods),
        "--quinn-policies", ",".join(policies),
        "--frames", str(args.frames or (299 if args.progressive else 30)),
        "--reps", str(args.reps or (1 if args.progressive else 3)),
        "--fps", str(args.fps or (1 if args.progressive else 30)),
        "--rate-mbps", str(args.rate_mbps),
        "--delay-ms", str(args.delay_ms),
        "--loss-pct", str(args.loss_pct),
        "--tier-delays-ms", ",".join(map(str, values)),
        "--tier-selection", "full",
        *(["--live", "--progressive"] if args.progressive else []),
        *(["--show-window"] if args.show_window else []),
        *(["--save-stages"] if args.save_stages else []),
        *([] if args.no_netns else ["--netns"]),
        *(["--dry-run"] if args.dry_run else []),
        *extra,
    ]
    subprocess.run(command, cwd=runner.parent, check=True)


if __name__ == "__main__":
    main()
