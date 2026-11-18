"""Small geometry helpers (quaternions, distances, intervals)."""
from __future__ import annotations

import numpy as np


def quat_to_euler_deg(q: np.ndarray):
    """q: (N,4) as [w,x,y,z] -> roll, pitch, yaw in degrees."""
    q = np.asarray(q, float)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)


def euler_to_quat(roll, pitch, yaw):
    """Radians -> (N,4) [w,x,y,z] (ZYX convention, same as PX4)."""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.column_stack([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def haversine_m(lat1, lon1, lat2, lon2):
    r = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dl = p2 - p1, np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def runs(mask: np.ndarray, t: np.ndarray, min_dur: float = 0.0, merge_gap: float = 0.0):
    """Turn a boolean mask into [(t_start, t_end), ...] intervals."""
    mask = np.asarray(mask, bool)
    if not mask.any():
        return []
    edges = np.diff(np.concatenate([[0], mask.view(np.int8), [0]]))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0] - 1
    out: list[list[float]] = []
    for s, e in zip(starts, ends):
        a, b = float(t[s]), float(t[e])
        if out and a - out[-1][1] <= merge_gap:
            out[-1][1] = b
        else:
            out.append([a, b])
    return [(a, b) for a, b in out if b - a >= min_dur]
