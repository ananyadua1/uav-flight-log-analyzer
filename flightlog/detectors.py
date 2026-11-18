"""Automatic problem detection: vibration, battery sag, GPS, tracking, dropouts."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
from scipy import signal, stats

from .maths import haversine_m, runs
from .parser import FlightLog


@dataclass
class Config:
    vib_warn: float = 3.0         # m/s^2 RMS (high-passed accel)
    vib_crit: float = 6.0
    att_err_warn: float = 10.0    # degrees
    att_err_crit: float = 45.0
    att_min_dur: float = 0.5      # s
    cells: int = 0                # 0 = auto-detect
    cell_warn: float = 3.5        # V per cell
    cell_crit: float = 3.3
    sag_per_cell: float = 0.15    # V/cell below running baseline
    capacity_mah: float = 0.0     # fallback for flight-time estimate
    gps_min_fix: int = 3
    gap_factor: float = 5.0       # gap if dt > factor * median dt
    gap_min_s: float = 0.25


@dataclass
class Issue:
    kind: str       # vibration | battery | gps | attitude | dropout | crash
    severity: str   # info | warning | critical
    t_start: float
    t_end: float
    detail: str

    def asdict(self):
        return asdict(self)


@dataclass
class Analysis:
    issues: list[Issue] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    series: dict = field(default_factory=dict)   # derived series for plotting
    score: int = 100

    def by_kind(self, kind):
        return [i for i in self.issues if i.kind == kind]


# --------------------------------------------------------------------------
def detect_vibration(log: FlightLog, cfg: Config, res: Analysis):
    if not log.has("accel"):
        res.metrics["vibration"] = None
        return
    df = log.streams["accel"]
    t = df["t"].to_numpy()
    fs = 1.0 / np.median(np.diff(t))
    if fs < 50:
        res.issues.append(Issue("vibration", "info", t[0], t[-1],
                                f"Accelerometer logged at only {fs:.0f} Hz: FFT check skipped."))
        return
    grid = np.arange(t[0], t[-1], 1 / fs)
    acc = np.column_stack([np.interp(grid, t, df[c]) for c in ("ax", "ay", "az")])
    hp = signal.butter(4, 5.0, "highpass", fs=fs, output="sos")
    vib = signal.sosfiltfilt(hp, acc, axis=0)          # remove gravity + manoeuvres
    mag2 = (vib ** 2).sum(axis=1)

    win = int(fs)                                        # 1 s windows
    n = len(mag2) // win
    rms = np.sqrt(mag2[: n * win].reshape(n, win).mean(axis=1))
    tw = grid[: n * win : win] + 0.5
    res.series["vib_rms"] = pd.DataFrame({"t": tw, "rms": rms})

    f, pxx = signal.welch(vib, fs=fs, nperseg=min(1024, len(vib)), axis=0)
    pxx = pxx.sum(axis=1)
    res.series["psd"] = pd.DataFrame({"f": f, "psd": pxx})
    band = f >= 5
    peak_f = float(f[band][np.argmax(pxx[band])])

    res.metrics["vibration"] = {
        "rms_mean": float(rms.mean()), "rms_max": float(rms.max()),
        "peak_freq_hz": peak_f, "sample_rate_hz": float(fs),
    }
    for lvl, thr in (("critical", cfg.vib_crit), ("warning", cfg.vib_warn)):
        for a, b in runs(rms > thr, tw, merge_gap=1.5):
            seg = rms[(tw >= a) & (tw <= b)]
            if any(i.kind == "vibration" and i.t_start <= a <= i.t_end for i in res.issues):
                continue
            res.issues.append(Issue("vibration", lvl, a, b,
                f"High vibration: RMS up to {seg.max():.1f} m/s² (limit {thr}); "
                f"FFT peak at {peak_f:.0f} Hz."))


def detect_battery(log: FlightLog, cfg: Config, res: Analysis):
    if not log.has("battery"):
        res.metrics["battery"] = None
        return
    df = log.streams["battery"].dropna(subset=["v"])
    df = df[df["v"] > 1.0].reset_index(drop=True)
    if len(df) < 10:
        res.metrics["battery"] = None
        return
    t, v, i = df["t"].to_numpy(), df["v"].to_numpy(), df["i"].to_numpy()
    cells = cfg.cells or int(np.ceil(np.nanmax(v) / 4.25))
    cells = max(cells, 1)
    dt = np.median(np.diff(t))
    k = max(3, int(10 / dt) | 1)                         # ~10 s running median
    v_base = pd.Series(v).rolling(k, center=True, min_periods=1).median().to_numpy()
    i_base = pd.Series(i).rolling(k, center=True, min_periods=1).median().to_numpy()
    dv, di = v - v_base, i - i_base

    # internal resistance from V-vs-I fluctuations (slope of dV/dI)
    r_int = None
    if np.nanstd(di) > 0.3:
        sl = stats.linregress(di, dv)
        if sl.slope < 0 and sl.rvalue ** 2 > 0.1:
            r_int = float(-sl.slope)

    # flight-time estimate
    t_left, method = None, None
    rem = df["remaining"].to_numpy()
    if np.isfinite(rem).sum() > 10 and np.nanmax(rem) <= 1.0 and np.nanmax(rem) - np.nanmin(rem) > 1e-3:
        s = stats.linregress(t[np.isfinite(rem)], rem[np.isfinite(rem)])
        if s.slope < 0:
            t_left, method = float(max(rem[np.isfinite(rem)][-1], 0) / -s.slope), "battery remaining % trend"
    if t_left is None and cfg.capacity_mah > 0 and np.isfinite(df["mah"]).any():
        used = float(np.nanmax(df["mah"]))
        avg_i = float(np.nanmean(i[t > t[-1] - 30]))
        if avg_i > 0.5:
            t_left, method = max(cfg.capacity_mah - used, 0) / (avg_i * 1000 / 3600), "capacity / recent current"

    cell_v = v / cells
    res.series["cell_v"] = pd.DataFrame({"t": t, "cell_v": cell_v, "sag": -dv / cells})
    res.metrics["battery"] = {
        "cells": cells, "v_start": float(v[:5].mean()), "v_end": float(v[-5:].mean()),
        "v_min": float(v.min()), "cell_min": float(cell_v.min()),
        "i_max": float(np.nanmax(i)), "i_mean": float(np.nanmean(i)),
        "r_int_ohm": r_int, "t_left_s": t_left, "t_left_method": method,
        "sag_max_cell": float((-dv / cells).max()),
    }
    sag_runs = runs(-dv / cells > cfg.sag_per_cell, t, merge_gap=2.0)
    if sag_runs:
        worst = float((-dv).max())
        a, b = sag_runs[0][0], sag_runs[-1][1]
        res.issues.append(Issue("battery", "warning", a, b,
            f"Battery sag: {len(sag_runs)} event(s), worst drop {worst:.2f} V below baseline "
            f"(peak {i.max():.0f} A)" + (f", R≈{r_int * 1000:.0f} mΩ." if r_int else ".")))
    for lvl, thr in (("critical", cfg.cell_crit), ("warning", cfg.cell_warn)):
        rr = runs(cell_v < thr, t, merge_gap=2.0)
        if rr and not (lvl == "warning" and any(x.kind == "battery" and x.severity == "critical" and "cell" in x.detail for x in res.issues)):
            m = cell_v < thr
            res.issues.append(Issue("battery", lvl, rr[0][0], rr[-1][1],
                f"Low cell voltage: {cell_v.min():.2f} V/cell minimum, below {thr} V in {len(rr)} period(s) "
                f"({m.mean() * 100:.0f}% of flight)."))


def detect_gps(log: FlightLog, cfg: Config, res: Analysis):
    if not log.has("gps"):
        res.metrics["gps"] = None
        return
    df = log.streams["gps"]
    t = df["t"].to_numpy()
    lost = df["fix"].to_numpy() < cfg.gps_min_fix
    n = 0
    for a, b in runs(lost, t, merge_gap=0.5):
        n += 1
        # extend to the next valid sample so the dropout covers the whole outage
        nxt = t[t > b]
        b2 = float(nxt[0]) if len(nxt) else b
        res.issues.append(Issue("gps", "critical" if b2 - a > 5 else "warning", a, b2,
                                f"GPS fix lost for {b2 - a:.1f} s (fix type < {cfg.gps_min_fix})."))
    gap_thr = max(1.0, cfg.gap_factor * np.median(np.diff(t)))
    for k in np.where(np.diff(t) > gap_thr)[0]:
        res.issues.append(Issue("gps", "warning", float(t[k]), float(t[k + 1]),
                                f"No GPS messages for {t[k + 1] - t[k]:.1f} s."))
    lat, lon = df["lat"].to_numpy(), df["lon"].to_numpy()
    good = ~lost & (lat != 0)
    dist = float(haversine_m(lat[good][:-1], lon[good][:-1], lat[good][1:], lon[good][1:]).sum()) if good.sum() > 2 else 0.0
    res.metrics["gps"] = {"dropouts": n, "min_sats": int(df["sats"].min()),
                          "max_sats": int(df["sats"].max()), "distance_m": dist}


def detect_attitude(log: FlightLog, cfg: Config, res: Analysis):
    if not (log.has("attitude") and log.has("attitude_sp")):
        res.metrics["attitude"] = None
        return
    a, s = log.streams["attitude"], log.streams["attitude_sp"]
    t = a["t"].to_numpy()
    q = a[["q0", "q1", "q2", "q3"]].to_numpy()
    qs = s[["q0", "q1", "q2", "q3"]].to_numpy()
    flip = np.concatenate([[1], np.cumprod(np.sign((qs[1:] * qs[:-1]).sum(1) + 1e-12))])
    qs = qs * flip[:, None]
    qd = np.column_stack([np.interp(t, s["t"], qs[:, c]) for c in range(4)])
    qd /= np.linalg.norm(qd, axis=1, keepdims=True)
    err = np.degrees(2 * np.arccos(np.clip(np.abs((q * qd).sum(1)), 0, 1)))
    dt = np.median(np.diff(t))
    err_s = pd.Series(err).rolling(max(1, int(0.3 / dt)), center=True, min_periods=1).median().to_numpy()
    res.series["att_err"] = pd.DataFrame({"t": t, "err": err_s})
    res.metrics["attitude"] = {"err_mean": float(err_s.mean()), "err_max": float(err_s.max()),
                               "err_p95": float(np.percentile(err_s, 95))}
    for lvl, thr in (("critical", cfg.att_err_crit), ("warning", cfg.att_err_warn)):
        for x, y in runs(err_s > thr, t, min_dur=cfg.att_min_dur if lvl == "warning" else 0.2, merge_gap=0.5):
            if any(i.kind in ("attitude", "crash") and x <= i.t_end and y >= i.t_start for i in res.issues):
                continue
            seg = err_s[(t >= x) & (t <= y)]
            kind = "crash" if lvl == "critical" else "attitude"
            msg = (f"Attitude diverged up to {seg.max():.0f}° from commanded: loss of control / possible crash."
                   if lvl == "critical" else
                   f"Attitude tracking error up to {seg.max():.0f}° (limit {thr:.0f}°) for {y - x:.1f} s.")
            res.issues.append(Issue(kind, lvl, x, y, msg))


def detect_dropouts(log: FlightLog, cfg: Config, res: Analysis):
    found = {}
    for name, df in log.streams.items():
        if name in ("gps", "attitude_sp") or len(df) < 10:
            continue
        t = df["t"].to_numpy()
        dts = np.diff(t)
        thr = max(cfg.gap_factor * np.median(dts), cfg.gap_min_s)
        for k in np.where(dts > thr)[0]:
            gap = float(dts[k])
            found.setdefault(name, 0)
            found[name] += 1
            res.issues.append(Issue("dropout", "critical" if gap > 2 else "warning",
                                    float(t[k]), float(t[k + 1]),
                                    f"Sensor dropout in '{name}': {gap:.1f} s gap "
                                    f"(normal period {np.median(dts) * 1000:.0f} ms)."))
    res.metrics["dropouts"] = found


def detect_motors(log: FlightLog, cfg: Config, res: Analysis):
    if not log.has("motors"):
        return
    m = log.streams["motors"]
    cols = [c for c in m.columns if c.startswith("m")]
    vals = m[cols].to_numpy()
    res.metrics["motors"] = {c: {"mean": float(m[c].mean()), "max": float(m[c].max())} for c in cols}
    hi = np.nanmax(vals)
    if hi > 100:  # PWM microseconds
        sat = (vals >= 0.97 * hi).any(axis=1) if hi > 1500 else np.zeros(len(m), bool)
        res.metrics["motor_saturation_pct"] = float(sat.mean() * 100)


def analyze(log: FlightLog, cfg: Config | None = None) -> Analysis:
    cfg = cfg or Config()
    res = Analysis()
    for fn in (detect_vibration, detect_battery, detect_gps, detect_attitude,
               detect_dropouts, detect_motors):
        fn(log, cfg, res)
    res.issues.sort(key=lambda i: i.t_start)
    if log.has("altitude"):
        res.metrics["max_alt_m"] = float(log.streams["altitude"]["alt"].max())
    res.metrics["duration_s"] = log.duration
    pen = {"info": 0, "warning": 6, "critical": 18}
    per_kind: dict[str, int] = {}
    for i in res.issues:
        per_kind[i.kind] = per_kind.get(i.kind, 0) + pen[i.severity]
    res.score = int(max(0, 100 - sum(min(v, 40) for v in per_kind.values())))
    return res
