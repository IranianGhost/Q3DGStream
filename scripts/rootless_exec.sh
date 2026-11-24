#!/usr/bin/env bash
set -euo pipefail
[[ $(uname -s) == Linux ]] || { echo 'GPU server Linux only' >&2; exit 64; }
if [[ ${1:-} != --inside ]]; then
    (( $# > 0 )) || { echo 'usage: rootless_exec.sh COMMAND [ARGS]' >&2; exit 64; }
    export QS_PARENT_NETNS=$(readlink /proc/self/ns/net)
    export QS_PARENT_MNTNS=$(readlink /proc/self/ns/mnt)
    export QS_PARENT_USERNS=$(readlink /proc/self/ns/user)
    exec unshare -Urnm bash "$0" --inside "$@"
fi
shift
[[ $(id -u) == 0 && -n ${QS_PARENT_NETNS:-} && -n ${QS_PARENT_MNTNS:-} && -n ${QS_PARENT_USERNS:-} ]] || exit 65
[[ $(readlink /proc/self/ns/net) != "$QS_PARENT_NETNS" && $(readlink /proc/self/ns/mnt) != "$QS_PARENT_MNTNS" && $(readlink /proc/self/ns/user) != "$QS_PARENT_USERNS" ]] || exit 65
read -r mapped inside_uid range < /proc/self/uid_map
[[ $mapped == 0 && $range == 1 ]] || exit 65
root=$(cd "$(dirname "$0")" && pwd)
runtime=$(mktemp -d "${TMPDIR:-/tmp}/q3dg-netns.XXXXXX")
mkdir -p "$runtime/run/netns"
cleanup() {
    # Every namespace alias visible here belongs to this private /run mount.
    while read -r ns rest; do [[ -z $ns ]] || ip netns del "$ns" >/dev/null 2>&1 || true; done < <(ip netns list)
    umount /run/netns >/dev/null 2>&1 || true
    umount /run >/dev/null 2>&1 || true
    rm -rf -- "$runtime"
}
mount --bind "$runtime/run" /run
trap cleanup EXIT
export QS_REPRO_ROOTLESS_READY=1
export PATH="$root/rootless-bin:$PATH"
"$@"
