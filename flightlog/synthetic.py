"""Generate realistic synthetic PX4 .ulg flight logs (real binary ULog format).

Used for the bundled sample logs and tests, so the project works with no
downloads. Real logs from review.px4.io load through the exact same parser.
"""
from __future__ import annotations

import struct
import zlib

import numpy as np

from .maths import euler_to_quat

LAT0, LON0 = 47.397742, 8.545594   # Zurich (PX4 SITL default home)
G = 9.81

SCHEMA = {
    "vehicle_local_position": [("uint64_t", "timestamp"), ("float", "x"), ("float", "y"), ("float", "z")],
    "vehicle_gps_position": [("uint64_t", "timestamp"), ("int32_t", "lat"), ("int32_t", "lon"),
                             ("int32_t", "alt"), ("uint8_t", "fix_type"), ("uint8_t", "satellites_used")],
    "vehicle_attitude": [("uint64_t", "timestamp"), ("float[4]", "q")],
    "vehicle_attitude_setpoint": [("uint64_t", "timestamp"), ("float[4]", "q_d")],
    "battery_status": [("uint64_t", "timestamp"), ("float", "voltage_v"), ("float", "current_a"),
                       ("float", "discharged_mah"), ("float", "remaining")],
    "actuator_outputs": [("uint64_t", "timestamp"), ("uint32_t", "noutputs"), ("float[8]", "output")],
    "sensor_combined": [("uint64_t", "timestamp"), ("float[3]", "gyro_rad"), ("float[3]", "accelerometer_m_s2")],
}
_CODE = {"uint64_t": "Q", "uint32_t": "I", "int32_t": "i", "uint8_t": "B", "float": "f"}


def _pack_fmt(schema):
    f = "<"
    for typ, _ in schema:
        base, _, n = typ.partition("[")
        f += (n[:-1] if n else "") + _CODE[base]
    return f


def write_ulog(path: str, topics: dict[str, dict[str, np.ndarray]], info: dict | None = None):
    """topics[name] = {'timestamp': us array, field: (N,) or (N,k) array, ...}"""
    out = bytearray(b"ULog\x01\x12\x35\x01" + struct.pack("<Q", 0))

    def msg(kind: bytes, payload: bytes):
        out.extend(struct.pack("<HB", len(payload), kind[0]) + payload)

    msg(b"B", bytes(8) + bytes(8) + bytes(24))
    for k, v in (info or {}).items():
        key = f"char[{len(v)}] {k}".encode()
        msg(b"I", struct.pack("<B", len(key)) + key + v.encode())
    ids = {}
    for name, sch in SCHEMA.items():
        if name in topics:
            msg(b"F", (f"{name}:" + "".join(f"{t} {n};" for t, n in sch)).encode())
    for mid, name in enumerate(n for n in SCHEMA if n in topics):
        ids[name] = mid
        msg(b"A", struct.pack("<BH", 0, mid) + name.encode())

    rows = []
    for name, cols in topics.items():
        sch, fmt = SCHEMA[name], _pack_fmt(SCHEMA[name])
        n = len(cols["timestamp"])
        for r in range(n):
            vals = []
            for typ, fname in sch:
                v = cols[fname][r]
                vals.extend(np.atleast_1d(v).tolist())
            rows.append((int(cols["timestamp"][r]), ids[name], struct.pack(fmt, *vals)))
    rows.sort(key=lambda x: x[0])
    for _, mid, payload in rows:
        msg(b"D", struct.pack("<H", mid) + payload)
    with open(path, "wb") as fh:
        fh.write(out)


# --------------------------------------------------------------------------
SCENARIOS = {
    "01_normal_flight": "Clean circular mission: nothing should be flagged.",
    "02_vibration_issue": "Unbalanced prop: strong ~70 Hz vibration for most of the flight.",
    "03_battery_sag": "Tired pack with high internal resistance: heavy sag and low cell voltage.",
    "04_gps_dropout": "Two GPS fix losses (jamming/multipath) mid-flight.",
    "05_sensor_dropouts": "IMU and battery data gaps (logging/brown-out problems).",
    "06_attitude_tracking": "Wind gust: the drone fails to follow commanded roll/pitch.",
    "07_crash_motor_failure": "Motor 3 fails at t=70 s: loss of control and impact.",
}


def generate(scenario: str, seed: int = 0, T: float = 150.0):
    rng = np.random.default_rng(seed + zlib.crc32(scenario.encode()) % 1000)

    def grid(hz):
        return np.arange(0, T, 1 / hz)

    # ---- true trajectory ----------------------------------------------
    R, v = 40.0, 5.0
    w = v / R

    def traj(t):
        circ = np.clip(t - 20, 0, 100)
        th = w * circ
        n, e = R * np.sin(th), R * (1 - np.cos(th))
        alt = np.interp(t, [0, 5, 20, 120, 145, T], [0, 0, 30, 30, 0, 0])
        return n, e, alt, th

    crash_t = 70.0 if scenario.startswith("07") else None
    if crash_t:
        T = 82.0

    # ---- attitude --------------------------------------------------------
    ta = grid(50)
    ta = ta[ta < T]
    n, e, alt, th = traj(ta)
    roll_c = np.radians(3.6 + 2.0 * np.sin(0.7 * ta))
    pitch_c = np.radians(-3 + 1.5 * np.sin(0.5 * ta + 1))
    yaw_c = th + np.pi / 2
    roll_c = np.where((ta > 20) & (ta < 120), roll_c, 0.0)
    pitch_c = np.where((ta > 20) & (ta < 120), pitch_c, 0.0)
    q_sp = euler_to_quat(roll_c, pitch_c, yaw_c)
    r_a, p_a, y_a = roll_c.copy(), pitch_c.copy(), yaw_c.copy()
    r_a += np.radians(rng.normal(0, 0.4, len(ta)))
    p_a += np.radians(rng.normal(0, 0.4, len(ta)))
    if scenario.startswith("06"):
        for a, b, amp in ((50, 60, 22), (95, 102, 16)):
            m = (ta > a) & (ta < b)
            ph = np.sin(np.pi * (ta[m] - a) / (b - a))
            r_a[m] += np.radians(amp) * ph
            p_a[m] -= np.radians(amp * 0.6) * ph
    if crash_t:
        m = ta >= crash_t
        k = ta[m] - crash_t
        r_a[m] += np.radians(40 * k ** 1.6)
        p_a[m] += np.radians(15 * k ** 1.3)
        y_a[m] += 2.5 * k
    q = euler_to_quat(r_a, p_a, y_a)

    # ---- position / altitude --------------------------------------------
    tp = grid(20)
    tp = tp[tp < T]
    n_p, e_p, alt, _ = traj(tp)
    if crash_t:
        m = tp >= crash_t
        fall = np.clip(1 - 0.5 * 4.0 * (tp[m] - crash_t) ** 2 * 1.2 / 30, 0, 1)  # free-fall-ish
        alt[m] = 30 * fall
    z = -(alt + rng.normal(0, 0.05, len(tp)))

    # ---- GPS ---------------------------------------------------------------
    tg = grid(5)
    tg = tg[tg < T]
    n, e, alt_g, _ = traj(tg)
    lat = LAT0 + n / 111320.0
    lon = LON0 + e / (111320.0 * np.cos(np.radians(LAT0)))
    fix = np.full(len(tg), 3)
    sats = rng.integers(11, 15, len(tg))
    if scenario.startswith("04"):
        for a, b in ((45, 58), (90, 95)):
            m = (tg >= a) & (tg < b)
            fix[m], sats[m] = 0, 0
            lat[m], lon[m] = lat[~m][tg[~m] < a][-1], lon[~m][tg[~m] < a][-1]
    if crash_t:
        keep = tg < T
        lat, lon = lat[keep], lon[keep]
    alt_gps = alt_g + 488 + rng.normal(0, 0.3, len(tg))

    # ---- battery -----------------------------------------------------------
    tb = grid(10)
    tb = tb[tb < T]
    sag = scenario.startswith("03")
    r_pack = 0.12 if sag else 0.05
    cap = 1500.0 if sag else 5000.0
    _, _, altb, _ = traj(tb)
    climb = np.gradient(altb, tb)
    i = 11 + 9 * np.clip(climb, 0, None) + (6 if sag else 0) * (tb > 20) * (tb < 120)
    i = i + 0.8 * np.sin(1.3 * tb) + rng.normal(0, 0.4, len(tb))
    if sag:
        i += 7 * (np.sin(0.45 * tb) > 0.6)            # punch-outs
    i = np.where((tb < 5) | (tb > 145), 0.6, i)
    mah = np.cumsum(i) * (tb[1] - tb[0]) / 3.6
    rem = 1 - mah / cap
    cell = 3.5 + 0.7 * np.clip(rem, 0, 1) ** 0.8
    vb = 4 * cell - i * r_pack + rng.normal(0, 0.02, len(tb))
    if crash_t:
        i = np.where(tb > crash_t + 8, 0.2, i)

    # ---- motors (PWM us) -----------------------------------------------------
    tm = grid(50)
    tm = tm[tm < T]
    _, _, altm, _ = traj(tm)
    base = 1000 + 380 + 120 * np.clip(np.gradient(altm, tm), 0, None) / 1.0 * 0.4
    base = np.where((tm < 5) | (tm > 145), 1100, base)
    mot = np.column_stack([base + rng.normal(0, 12, len(tm)) + 25 * np.sin(2.1 * tm + k) for k in range(4)])
    if crash_t:
        m = tm >= crash_t
        mot[m, 2] = 1000
        mot[m, 0] = mot[m, 1] = np.clip(mot[m, 0] + 500, 1000, 2000)
        mot[m, 3] = 2000
        mot[tm > crash_t + 7] = 1000
    mot = np.clip(mot, 1000, 2000)
    out8 = np.zeros((len(tm), 8))
    out8[:, :4] = mot

    # ---- IMU ---------------------------------------------------------------------
    ts = grid(200)
    ts = ts[ts < T]
    acc = np.zeros((len(ts), 3))
    acc[:, 2] = -G
    acc += rng.normal(0, 0.25, acc.shape)
    acc[:, 0] += 0.4 * np.sin(0.3 * ts)
    amp = np.zeros(len(ts))
    if scenario.startswith("02"):
        amp = np.where((ts > 15) & (ts < 130), 6.5, 0.0)
        amp *= 1 + 0.15 * np.sin(0.8 * ts)
    elif scenario.startswith("01") or True:
        amp = np.where((ts > 5) & (ts < 145), 0.35, 0.0)
    f_vib = 70.0
    acc[:, 0] += amp * np.sin(2 * np.pi * f_vib * ts)
    acc[:, 1] += 0.8 * amp * np.sin(2 * np.pi * f_vib * ts + 1)
    acc[:, 2] += 1.2 * amp * np.sin(2 * np.pi * f_vib * ts + 2)
    if crash_t:
        acc[(ts > crash_t + 8) & (ts < crash_t + 8.15)] += rng.normal(0, 40, (30, 3))[: np.sum((ts > crash_t + 8) & (ts < crash_t + 8.15))]
    gyro = rng.normal(0, 0.02, (len(ts), 3))

    keep_s = np.ones(len(ts), bool)
    keep_b = np.ones(len(tb), bool)
    if scenario.startswith("05"):
        keep_s &= ~(((ts > 60) & (ts < 64.2)) | ((ts > 105) & (ts < 106.5)))
        keep_b &= ~((tb > 80) & (tb < 83.5))

    def us(t):
        return (np.asarray(t) * 1e6).astype(np.uint64) + np.uint64(2_000_000)

    topics = {
        "vehicle_attitude": {"timestamp": us(ta), "q": q},
        "vehicle_attitude_setpoint": {"timestamp": us(ta), "q_d": q_sp},
        "vehicle_local_position": {"timestamp": us(tp), "x": n_p, "y": e_p, "z": z},
        "vehicle_gps_position": {"timestamp": us(tg), "lat": (lat * 1e7).astype(np.int32),
                                 "lon": (lon * 1e7).astype(np.int32), "alt": (alt_gps * 1e3).astype(np.int32),
                                 "fix_type": fix.astype(np.uint8), "satellites_used": sats.astype(np.uint8)},
        "battery_status": {"timestamp": us(tb[keep_b]), "voltage_v": vb[keep_b], "current_a": i[keep_b],
                           "discharged_mah": mah[keep_b], "remaining": rem[keep_b]},
        "actuator_outputs": {"timestamp": us(tm), "noutputs": np.full(len(tm), 4, np.uint32), "output": out8},
        "sensor_combined": {"timestamp": us(ts[keep_s]), "gyro_rad": gyro[keep_s], "accelerometer_m_s2": acc[keep_s]},
    }
    # crash: truncate every topic at the end of the log
    if crash_t:
        for name, cols in topics.items():
            m = cols["timestamp"] < us(np.array([T]))[0]
            topics[name] = {k: v[m] for k, v in cols.items()}
    return topics


def make_all(outdir: str):
    import os
    os.makedirs(outdir, exist_ok=True)
    paths = []
    for k, (name, desc) in enumerate(SCENARIOS.items()):
        p = os.path.join(outdir, f"{name}.ulg")
        write_ulog(p, generate(name, seed=k), info={"sys_name": "PX4", "ver_sw": "synthetic-1.0", "desc": desc})
        paths.append(p)
    return paths
