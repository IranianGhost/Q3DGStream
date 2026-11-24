"""Pure NumPy scoring and schedule helpers; imported by GPU scheduler and CPU tests."""
from __future__ import annotations
import numpy as np

TIER_WEIGHTS = np.asarray([12.0, 4.0, 1.0, 0.2], dtype=np.float64)


def interpolate_saliency(timeline: dict, frame: int) -> np.ndarray:
    anchors = np.asarray(timeline['frames'], dtype=np.float64)
    scores = np.asarray(timeline['scores'], dtype=np.float64)
    n = int(timeline['tiles'])
    if anchors.ndim != 1 or len(anchors) < 2 or not np.all(np.diff(anchors) > 0):
        raise ValueError('Need >=2 sorted, unique saved saliency anchors')
    if scores.shape != (len(anchors), n) or not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError('Invalid timeline score shape or values')
    return np.array([np.interp(frame, anchors, scores[:, t]) for t in range(n)], dtype=np.float64)


def priority_tiers(sal: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Exactly the previously established P0/P1/nearby P2/P3 tile rule."""
    n = len(sal)
    if n < 8 or centroids.shape != (n, 3):
        raise ValueError('Invalid tile structure')
    if not np.isfinite(sal).all() or (sal < 0).any() or float(np.max(sal)) <= 0:
        raise ValueError('Saliency scores missing or all zero; no silent fallback')
    ranked = np.argsort(-sal, kind='stable')
    selected = ranked[:4]
    rest = np.asarray([t for t in range(n) if t not in selected], dtype=np.int64)
    near_dist = np.linalg.norm(centroids[rest,None,:] - centroids[selected][None,:,:],axis=-1).min(axis=1)
    near = rest[np.argsort(near_dist, kind='stable')[:4]]
    tiers = np.full(n, 3, dtype=np.uint8)
    tiers[ranked[:2]] = 0
    tiers[ranked[2:4]] = 1
    tiers[near] = 2
    return tiers


def rank_retained_blocks(block_ids: np.ndarray, tile_sens: np.ndarray,
                         tiers: np.ndarray, saliency: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rank original retained HashGrid IDs, MLP IDs first; return ALL IDs, raw priority scores.

    The tile_sens matrix must be computed anew for each checkpoint from its
    decoded wire values. No assumptions that block memberships match NTC114.
    """
    ids = np.asarray(block_ids, dtype=np.int64)
    if ids.ndim != 1 or len(np.unique(ids)) != len(ids) or (ids < 0).any():
        raise ValueError('Invalid retained block IDs')
    if tile_sens.shape != (len(tiers), len(ids)):
        raise ValueError('Sensitivity shape mismatch')
    if not np.isfinite(tile_sens).all() or (tile_sens < 0).any():
        raise ValueError('Non-finite/negative sensitivity')
    if saliency.shape != tiers.shape:
        raise ValueError('Saliency/tier shape mismatch')
    # Existing four-tier rule; the changing saliency confidence also contributes.
    sal_normalized = saliency / max(float(np.max(saliency)), 1e-12)
    weights = TIER_WEIGHTS[tiers] * (0.2 + 0.8 * sal_normalized)
    values = (weights[:, None] * tile_sens).sum(axis=0)
    # Original retained position is the deterministic tiebreaker.
    order = np.argsort(-values, kind='stable')
    return ids[order].astype(np.int32), values[order].astype(np.float64)


def partition_in_four(hash_ids: np.ndarray, tail_ids: np.ndarray) -> list[np.ndarray]:
    n = len(hash_ids)
    bounds = [0, round(n*.25), round(n*.50), round(n*.75), n]
    return [np.concatenate([tail_ids, hash_ids[:bounds[1]]]).astype(np.int32)] + [
        hash_ids[bounds[k]:bounds[k+1]].astype(np.int32) for k in (1,2,3)]
