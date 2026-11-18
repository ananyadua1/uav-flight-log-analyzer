"""Read PX4 .ulg files with pyulog and return tidy Pandas DataFrames.

All DataFrames use a ``t`` column = seconds since log start.
Topic names differ between PX4 versions, so every stream has fallbacks.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from typing import BinaryIO

import numpy as np
import pandas as pd
from pyulog import ULog

from .maths import quat_to_euler_deg


@dataclass
class FlightLog:
    name: str
    duration: float
    streams: dict[str, pd.DataFrame] = field(default_factory=dict)
    info: dict = field(default_factory=dict)
    topics: list[str] = field(default_factory=list)

    def has(self, key: str) -> bool:
        return key in self.streams and len(self.streams[key]) > 1


def _dataset(ulog: ULog, *names: str):
    for n in names:
        try:
            ds = ulog.get_dataset(n)
            if len(ds.data["timestamp"]) > 0:
                return ds.data
        except (KeyError, IndexError, ValueError):
            continue
    return None


def _cols(data: dict, prefix: str) -> list[str]:
    """Columns like 'q[0]','q[1]' in numeric order."""
    cols = [k for k in data if k.startswith(prefix + "[")]
    return sorted(cols, key=lambda c: int(c[len(prefix) + 1 : -1]))


def _time(ulog: ULog, data: dict) -> np.ndarray:
    start = ulog.start_timestamp or int(data["timestamp"][0])
    return (np.asarray(data["timestamp"], dtype=np.int64) - start) / 1e6


def load_ulog(source: str | bytes | BinaryIO, name: str = "log") -> FlightLog:
    """Parse a .ulg from a path, raw bytes or an uploaded file object."""
    tmp = None
    try:
        if isinstance(source, str):
            path = source
            name = os.path.basename(source)
        else:
            raw = source if isinstance(source, (bytes, bytearray)) else source.read()
            fd, tmp = tempfile.mkstemp(suffix=".ulg")
            with os.fdopen(fd, "wb") as fh:
                fh.write(raw)
            path = tmp
        ulog = ULog(path)
    finally:
        if tmp and os.path.exists(tmp):
            os.remove(tmp)

    log = FlightLog(name=name, duration=0.0)
    log.topics = sorted({d.name for d in ulog.data_list})
    log.info = {k: v for k, v in ulog.msg_info_dict.items() if isinstance(v, (str, int, float))}
    end = ulog.last_timestamp or 0
    start = ulog.start_timestamp or 0
    log.duration = max(0.0, (end - start) / 1e6)

    # ---- altitude (relative, metres up) ----------------------------------
    d = _dataset(ulog, "vehicle_local_position")
    if d is not None:
        log.streams["altitude"] = pd.DataFrame(
            {"t": _time(ulog, d), "alt": -np.asarray(d["z"], float),
             "x": np.asarray(d["x"], float), "y": np.asarray(d["y"], float)}
        )
    else:
        d = _dataset(ulog, "vehicle_global_position")
        if d is not None:
            log.streams["altitude"] = pd.DataFrame(
                {"t": _time(ulog, d), "alt": np.asarray(d["alt"], float)})

    # ---- GPS -------------------------------------------------------------
    d = _dataset(ulog, "vehicle_gps_position", "sensor_gps")
    if d is not None:
        lat = np.asarray(d["lat"], float) * 1e-7
        lon = np.asarray(d["lon"], float) * 1e-7
        n = len(lat)
        df = pd.DataFrame({
            "t": _time(ulog, d), "lat": lat, "lon": lon,
            "alt": np.asarray(d["alt"], float) * 1e-3,
            "fix": np.asarray(d["fix_type"], int),
            "sats": np.asarray(d["satellites_used"], int) if "satellites_used" in d else np.zeros(n, int),
        })
        log.streams["gps"] = df

    # ---- attitude --------------------------------------------------------
    d = _dataset(ulog, "vehicle_attitude")
    if d is not None:
        q = np.column_stack([np.asarray(d[c], float) for c in _cols(d, "q")])
        r, p, y = quat_to_euler_deg(q)
        df = pd.DataFrame({"t": _time(ulog, d), "roll": r, "pitch": p, "yaw": y})
        for i in range(4):
            df[f"q{i}"] = q[:, i]
        log.streams["attitude"] = df
    d = _dataset(ulog, "vehicle_attitude_setpoint")
    if d is not None:
        q = np.column_stack([np.asarray(d[c], float) for c in _cols(d, "q_d")])
        r, p, y = quat_to_euler_deg(q)
        df = pd.DataFrame({"t": _time(ulog, d), "roll": r, "pitch": p, "yaw": y})
        for i in range(4):
            df[f"q{i}"] = q[:, i]
        log.streams["attitude_sp"] = df

    # ---- battery ---------------------------------------------------------
    d = _dataset(ulog, "battery_status")
    if d is not None:
        n = len(d["timestamp"])
        log.streams["battery"] = pd.DataFrame({
            "t": _time(ulog, d),
            "v": np.asarray(d["voltage_v"], float),
            "i": np.asarray(d["current_a"], float),
            "mah": np.asarray(d.get("discharged_mah", np.full(n, np.nan)), float),
            "remaining": np.asarray(d.get("remaining", np.full(n, np.nan)), float),
        })

    # ---- motors ----------------------------------------------------------
    d = _dataset(ulog, "actuator_outputs")
    prefix = "output"
    if d is None:
        d = _dataset(ulog, "actuator_motors")
        prefix = "control"
    if d is not None:
        df = pd.DataFrame({"t": _time(ulog, d)})
        for i, c in enumerate(_cols(d, prefix)):
            v = np.asarray(d[c], float)
            if np.nanmax(np.abs(np.nan_to_num(v))) > 0:
                df[f"m{i + 1}"] = v
        if df.shape[1] > 1:
            log.streams["motors"] = df

    # ---- IMU (for vibration) --------------------------------------------
    d = _dataset(ulog, "sensor_combined")
    if d is not None and _cols(d, "accelerometer_m_s2"):
        a = np.column_stack([np.asarray(d[c], float) for c in _cols(d, "accelerometer_m_s2")])
        log.streams["accel"] = pd.DataFrame(
            {"t": _time(ulog, d), "ax": a[:, 0], "ay": a[:, 1], "az": a[:, 2]})
    else:
        d = _dataset(ulog, "vehicle_acceleration")
        if d is not None and _cols(d, "xyz"):
            a = np.column_stack([np.asarray(d[c], float) for c in _cols(d, "xyz")])
            log.streams["accel"] = pd.DataFrame(
                {"t": _time(ulog, d), "ax": a[:, 0], "ay": a[:, 1], "az": a[:, 2]})

    for k, df in log.streams.items():
        df.sort_values("t", inplace=True)
        df.reset_index(drop=True, inplace=True)
    return log
