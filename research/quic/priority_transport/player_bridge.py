"""Bridge committed network files to the existing player's checkpoint interface.

Complete-stage mode: publish an NTC only after all four QNP01 stages validate.
No partial transform is applied, so repeated refinements cannot accumulate motion.
"""

import json
import os
from pathlib import Path
import re
import tempfile
import threading

from q3dgstream import parse_qblk

STAGE = re.compile(r"^NTCs/progressive/NTC_(\d{6})_P([0-3])\.qblk$")


def ready_display_frame(ntcs, additions, total_frames, require_additions=True):
    frame = 0
    while frame in ntcs and (not require_additions or frame in additions):
        frame += 1
    return min(frame, max(0, total_frames - 1)) if total_frames else frame


class PlayerCacheBridge:
    def __init__(self, cache, notify):
        self.cache = Path(cache)
        self.notify = notify
        self.stages = {}
        self.finished = set()
        self.lock = threading.Lock()

    def committed(self, name):
        match = STAGE.fullmatch(name)
        if match is None:
            self.notify(name)
            return
        frame, stage = map(int, match.groups())
        decoded = parse_qblk((self.cache / name).read_bytes())
        if decoded[:3] != (frame, stage, 1556):
            raise ValueError("stage filename/header mismatch")
        with self.lock:
            parts = self.stages.setdefault(frame, {})
            if stage in parts or frame in self.finished:
                raise ValueError("duplicate stage")
            parts[stage] = name
            if len(parts) != 4:
                return
            import numpy as np
            import torch
            from research.quic.progressive_299_v2.qntc_progressive_wire import (
                decode_records_into,
                TOTAL,
                BLOCK,
            )

            params = np.zeros((TOTAL, BLOCK), np.float32)
            ids = set()
            for priority in range(4):
                _, _, _, records = parse_qblk(
                    (self.cache / parts[priority]).read_bytes()
                )
                incoming = {r[0] for r in records}
                if ids & incoming:
                    raise ValueError("duplicate blocks across stages")
                ids.update(incoming)
                decode_records_into(params, records)
            if len(ids) != 1556:
                raise ValueError("incomplete compressed NTC")
            bounds = json.loads(
                (self.cache / "NTCs/progressive/bounds.json").read_text()
            )
            lo = torch.tensor(bounds["xyz_bound_min"], dtype=torch.float32)
            hi = torch.tensor(bounds["xyz_bound_max"], dtype=torch.float32)
            if (
                lo.shape != (3,)
                or hi.shape != (3,)
                or not torch.isfinite(lo).all()
                or not torch.isfinite(hi).all()
                or not (hi > lo).all()
            ):
                raise ValueError("invalid NTC bounds")
            checkpoint = {
                "model.params": torch.from_numpy(params.reshape(-1)),
                "xyz_bound_min": lo,
                "xyz_bound_max": hi,
            }
            target = self.cache / f"NTCs/NTC_{frame:06d}.pth"
            if target.exists():
                raise ValueError("checkpoint already exists")
            fd, tmp = tempfile.mkstemp(dir=target.parent, suffix=".part")
            os.close(fd)
            try:
                torch.save(checkpoint, tmp)
                os.replace(tmp, target)
            finally:
                Path(tmp).unlink(missing_ok=True)
            self.finished.add(frame)
            del self.stages[frame]
            self.notify(target.relative_to(self.cache).as_posix())
