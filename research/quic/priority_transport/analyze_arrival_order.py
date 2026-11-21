"""Rank P0..P3 NTC group completion by receiver time for every treatment.

New runs distinguish fully reassembled QUIC data (GROUP_NET_READY) from the
later on-disk commit (GROUP_COMPLETE). Older CSVs are supported with a clearly
marked, disk-commit-only fallback.
"""

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median


def read_orders(source: Path):
    with source.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No per-frame records in {source}")
    out = []
    for row in rows:
        count = int(row["selected_tiers"])
        if not 1 <= count <= 4:
            raise ValueError(f"Invalid selected_tiers={count} in {source}")
        commit = {tier: float(row[f"p{tier}_complete_ms_connection"])
                  for tier in range(count)}
        has_network_times = all(row.get(f"p{tier}_net_ready_ms_connection")
                                for tier in range(count))
        network = ({tier: float(row[f"p{tier}_net_ready_ms_connection"])
                    for tier in range(count)} if has_network_times else commit)
        if has_network_times and any(network[tier] > commit[tier] for tier in network):
            raise ValueError("Receiver network-ready time exceeds disk-ready time")
        ordered_net = sorted(network, key=lambda tier: (network[tier], tier))
        ordered_disk = sorted(commit, key=lambda tier: (commit[tier], tier))
        first_strict = count == 1 or network[0] < min(network[t] for t in range(1, count))
        out.append({
            "repetition": int(row["repetition"]),
            "mode": row["mode"],
            "quinn_priority": row.get("quinn_priority") or row["quinn_priority_policy"],
            "frame": int(row["frame"]),
            "selected_tiers": count,
            "timestamp_source": "quic_reassembled" if has_network_times else "disk_commit_only",
            "net_ready_order": ">".join(f"P{tier}" for tier in ordered_net),
            "disk_ready_order": ">".join(f"P{tier}" for tier in ordered_disk),
            "net_and_disk_order_same": ordered_net == ordered_disk,
            "p0_first": first_strict,
            "tied_net_timestamps": len(set(network.values())) != count,
            "p0_net_rank": ordered_net.index(0) + 1,
            "net_ready_spread_ms": round(max(network.values()) - min(network.values()), 3),
            "disk_ready_spread_ms": round(max(commit.values()) - min(commit.values()), 3),
            **{f"p{tier}_net_ready_ms": network.get(tier, "") for tier in range(4)},
            **{f"p{tier}_disk_ready_ms": commit.get(tier, "") for tier in range(4)},
        })
    return out


def summarize(rows):
    treatments = defaultdict(list)
    for row in rows:
        treatments[(row["mode"], row["quinn_priority"])].append(row)
    summary = []
    for (mode, policy), group in sorted(treatments.items()):
        n = len(group)
        counts = Counter(row["net_ready_order"] for row in group)
        disk_counts = Counter(row["disk_ready_order"] for row in group)
        most_common_net, most_common_count = counts.most_common(1)[0]
        most_common_disk = disk_counts.most_common(1)[0][0]
        p0_first = sum(row["p0_first"] for row in group)
        summary.append({
            "mode": mode,
            "quinn_priority": policy,
            "ntc_frames": n,
            "quic_reassembly_timestamps": sum(
                row["timestamp_source"] == "quic_reassembled" for row in group),
            "disk_only_timestamps": sum(
                row["timestamp_source"] == "disk_commit_only" for row in group),
            "p0_first": p0_first,
            "p0_first_pct": round(100 * p0_first / n, 2),
            "p0_first_rank_median": median(row["p0_net_rank"] for row in group),
            "most_common_network_order": most_common_net,
            "most_common_network_order_count": most_common_count,
            "most_common_disk_order": most_common_disk,
            "median_net_ready_spread_ms": round(median(
                row["net_ready_spread_ms"] for row in group), 3),
            "median_disk_ready_spread_ms": round(median(
                row["disk_ready_spread_ms"] for row in group), 3),
            "network_disk_order_agreements": sum(
                row["net_and_disk_order_same"] for row in group),
            "tied_net_timestamps": sum(row["tied_net_timestamps"] for row in group),
        })
    return summary


def write_csv(path: Path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path,
                        help="Results directory or its per_frame_all.csv")
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    if source.is_dir():
        source /= "per_frame_all.csv"
    if not source.is_file():
        parser.error(f"Missing result CSV: {source}")
    per_frame = read_orders(source)
    summary = summarize(per_frame)
    output_dir = source.parent
    per_frame_out = output_dir / "arrival_order_per_frame.csv"
    summary_out = output_dir / "arrival_order_by_method.csv"
    write_csv(per_frame_out, per_frame)
    write_csv(summary_out, summary)
    print(f"ANALYZED {len(per_frame)} NTCs across {len(summary)} treatments")
    print(f"{'Application':20} {'Quinn':7} {'P0 first':>12}  {'Most common order':24} {'Median spread':>14}")
    for row in summary:
        first = f"{row['p0_first']}/{row['ntc_frames']}"
        print(f"{row['mode']:20} {row['quinn_priority']:7} {first:>12}  "
              f"{row['most_common_network_order']:24} {row['median_net_ready_spread_ms']:10.3f} ms")
    if any(row["timestamp_source"] == "disk_commit_only" for row in per_frame):
        print("CAUTION: Old runs have disk-commit timestamps only. These are NOT pure QUIC delivery times.")
    print(f"PER NTC  {per_frame_out}")
    print(f"SUMMARY  {summary_out}")


if __name__ == "__main__":
    main()
