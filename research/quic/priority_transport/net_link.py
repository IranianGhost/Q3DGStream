"""Temporary two-namespace shaped link; never changes an existing interface."""

from contextlib import contextmanager
import math
import os
import shutil
import subprocess
import sys
import time


WIFI_IMPAIRMENTS = {
    "campus": (),
    "delay": ("delay", "40ms"),
    "jitter": ("delay", "40ms", "10ms", "distribution", "normal"),
    "loss": ("delay", "20ms", "loss", "random", "2%"),
    "burst": ("delay", "20ms", "loss", "gemodel", "1%", "25%", "80%", "0.1%"),
}
QUEUE_PACKETS = WIFI_QUEUE_PACKETS = 1000


def privileged(*args):
    subprocess.run(["sudo", "-n", *args], check=True, stdout=subprocess.DEVNULL)


def parse_rate_profile(spec: str):
    """Parse elapsed-second:Mbps pairs, e.g. 0:20,3:5,6:10,9:20."""
    if not spec:
        return []
    points = []
    for part in spec.split(","):
        try:
            elapsed, mbps = map(float, part.split(":"))
        except (ValueError, TypeError):
            raise ValueError(f"Invalid bandwidth point {part!r}; use seconds:Mbps") from None
        if (not math.isfinite(elapsed) or not math.isfinite(mbps)
                or elapsed < 0 or mbps <= 0 or (points and elapsed <= points[-1][0])):
            raise ValueError("Bandwidth times must increase strictly and rates must be positive")
        points.append((elapsed, mbps))
    if not points or points[0][0] != 0:
        raise ValueError("Bandwidth profile must begin at t=0 seconds")
    return points


def apply_rate_profile(points, network, delay_ms, loss_pct, stop_event, epoch, errors,
                       forward_only=False, changes=None):
    """Change the bottleneck rate within the temporary link."""
    directions = [(network["client_ns"], network["client_dev"])]
    if not forward_only and "wifi_impairment" not in network:
        directions.append((network["server_ns"], network["server_dev"]))
    impairment = network.get("wifi_impairment")
    impairments = (WIFI_IMPAIRMENTS[impairment] if impairment is not None else
                   ["delay", f"{delay_ms:g}ms", "loss", f"{loss_pct:g}%"])
    netem = ["limit", str(QUEUE_PACKETS), *impairments]
    if changes is not None:
        changes.append({"scheduled_s": 0.0, "applied_s": 0.0, "rate_mbps": points[0][1]})
    try:
        for elapsed, mbps in points[1:]:
            remaining = epoch + elapsed - time.monotonic()
            if stop_event.wait(max(0, remaining)):
                return
            for ns, dev in directions:
                privileged("ip", "netns", "exec", ns, "tc", "qdisc", "change",
                           "dev", dev, "root", "netem", *netem, "rate", f"{mbps:g}mbit")
            applied = time.monotonic() - epoch
            if changes is not None:
                changes.append({"scheduled_s": elapsed, "applied_s": round(applied, 6), "rate_mbps": mbps})
            if changes is None or elapsed.is_integer():
                print(f"NETWORK_CHANGE at {applied:.3f}s -> {mbps:g}Mbps", flush=True)
    except Exception as exc:
        errors.append(str(exc))


@contextmanager
def isolated_wifi_loopback(rate_mbps, impairment):
    """Replay the campus rate and impair both directions on a shared queue."""
    subprocess.run(["sudo", "-v"], check=True)
    namespace = f"qntcn_wifi_{os.getpid()}"
    privileged("ip", "netns", "add", namespace)
    try:
        privileged("ip", "-n", namespace, "link", "set", "lo", "up")
        privileged("ip", "netns", "exec", namespace, "tc", "qdisc", "add",
                   "dev", "lo", "root", "netem", "limit", str(WIFI_QUEUE_PACKETS),
                   *WIFI_IMPAIRMENTS[impairment], "rate", f"{rate_mbps:g}mbit")
        print(f"NETWORK shared loopback {impairment}, initial rate {rate_mbps:g} Mbps", flush=True)
        yield {"host": "127.0.0.1", "client_ns": namespace, "server_ns": namespace,
               "client_dev": "lo", "server_dev": "lo", "wifi_impairment": impairment}
    finally:
        privileged("ip", "netns", "del", namespace)


@contextmanager
def isolated_link(rate_mbps, delay_ms, loss_pct, forward_only=False):
    for program in ("sudo", "ip", "tc"):
        if not shutil.which(program):
            raise RuntimeError(f"Missing {program} (ip/tc are in Ubuntu's iproute2)")
    subprocess.run(["sudo", "-v"], check=True)
    suffix = str(os.getpid())
    ns_server, ns_client = f"qntcn_s_{suffix}", f"qntcn_c_{suffix}"
    dev_server, dev_client = f"vns{suffix}"[:15], f"vnc{suffix}"[:15]
    address_server, address_client = "10.246.81.1", "10.246.81.2"
    output = subprocess.run(["sudo", "-n", "ip", "netns", "list"], check=True, text=True, capture_output=True).stdout
    existing = {line.split()[0] for line in output.splitlines() if line.split()}
    if ns_server in existing or ns_client in existing:
        raise RuntimeError("Temporary namespace name collision; nothing changed")
    for device in (dev_server, dev_client):
        if subprocess.run(["ip", "link", "show", "dev", device], capture_output=True).returncode == 0:
            raise RuntimeError(f"Interface {device} exists; nothing changed")

    created_server = created_client = created_veth = False
    try:
        privileged("ip", "netns", "add", ns_server)
        created_server = True
        privileged("ip", "netns", "add", ns_client)
        created_client = True
        privileged("ip", "link", "add", dev_server, "type", "veth", "peer", "name", dev_client)
        created_veth = True
        privileged("ip", "link", "set", dev_server, "netns", ns_server)
        privileged("ip", "link", "set", dev_client, "netns", ns_client)
        for ns, dev, addr in ((ns_server, dev_server, address_server), (ns_client, dev_client, address_client)):
            privileged("ip", "-n", ns, "addr", "add", addr + "/24", "dev", dev)
            privileged("ip", "-n", ns, "link", "set", "lo", "up")
            privileged("ip", "-n", ns, "link", "set", dev, "up")
            shaper = [] if forward_only and ns == ns_server else ["rate", f"{rate_mbps:g}mbit"]
            privileged("ip", "netns", "exec", ns, "tc", "qdisc", "add", "dev", dev, "root", "netem",
                       "limit", str(QUEUE_PACKETS), "delay", f"{delay_ms:g}ms",
                       "loss", f"{loss_pct:g}%", *shaper)
        print(f"NETWORK {rate_mbps:g}Mbps {'sender egress' if forward_only else 'per direction'} "
              f"RTT~{2 * delay_ms:g}ms loss={loss_pct:g}% per direction", flush=True)
        yield {"host": address_server, "server_ns": ns_server, "client_ns": ns_client,
               "server_dev": dev_server, "client_dev": dev_client}
    finally:
        if created_veth:
            for dev in (dev_server, dev_client):
                if subprocess.run(["ip", "link", "show", "dev", dev], capture_output=True).returncode == 0:
                    try:
                        privileged("ip", "link", "del", "dev", dev)
                    except subprocess.CalledProcessError:
                        print(f"WARNING: temporary veth {dev} remains", file=sys.stderr)
        for created, ns in ((created_client, ns_client), (created_server, ns_server)):
            if created:
                try:
                    privileged("ip", "netns", "del", ns)
                except subprocess.CalledProcessError:
                    print(f"WARNING: temporary namespace {ns} remains", file=sys.stderr)
