"""UAV Flight Log Analyzer & Telemetry Dashboard  (streamlit run app.py)"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from flightlog.detectors import Config, analyze
from flightlog.parser import load_ulog
from flightlog.report import ADVICE, ICON, build_html, build_markdown, grade
from flightlog.synthetic import SCENARIOS

HERE = os.path.dirname(os.path.abspath(__file__))
SEV_COLOR = {"critical": "rgba(239,68,68,0.25)", "warning": "rgba(245,158,11,0.22)", "info": "rgba(96,165,250,0.15)"}
LINE = ["#4cc9f0", "#f72585", "#b5e48c", "#ffd166", "#a78bfa", "#fb923c"]

st.set_page_config(page_title="UAV Flight Log Analyzer", page_icon="🛩️", layout="wide")


@st.cache_data(show_spinner="Parsing log…")
def parse(data: bytes, name: str):
    return load_ulog(data, name=name)


def shade(fig, issues, kinds=None, row=None):
    for i in issues:
        if kinds and i.kind not in kinds:
            continue
        kw = dict(x0=i.t_start, x1=max(i.t_end, i.t_start + 0.3), fillcolor=SEV_COLOR[i.severity],
                  line_width=0, layer="below")
        if row:
            fig.add_vrect(row=row, col=1, **kw)
        else:
            fig.add_vrect(**kw)


def style(fig, h=320, title=None):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=40 if title else 10, b=10), title=title,
                      legend=dict(orientation="h", y=1.12), hovermode="x unified")
    fig.update_xaxes(title_text="time (s)")
    return fig


# ---------------------------------------------------------------- sidebar --
st.sidebar.title("🛩️ Flight Log Analyzer")
src = st.sidebar.radio("Log source", ["Sample logs", "Upload .ulg"], horizontal=True)
data, name = None, None
if src == "Upload .ulg":
    up = st.sidebar.file_uploader("PX4 ULog file", type=["ulg"])
    st.sidebar.caption("Get real logs from [review.px4.io](https://review.px4.io) → Browse → Download.")
    if up:
        data, name = up.getvalue(), up.name
else:
    files = sorted(glob.glob(os.path.join(HERE, "samples", "*.ulg")))
    if files:
        paths = {os.path.basename(f)[:-4]: f for f in files}
        key = st.sidebar.selectbox("Sample", list(paths))
        pick = paths[key]
        st.sidebar.caption(SCENARIOS.get(key, ""))
        with open(pick, "rb") as fh:
            data, name = fh.read(), os.path.basename(pick)
    else:
        st.sidebar.warning("No samples found. Run `python scripts/generate_samples.py`.")

with st.sidebar.expander("Detection thresholds"):
    cfg = Config(
        vib_warn=st.number_input("Vibration warn (m/s² RMS)", 0.5, 50.0, 3.0, 0.5),
        vib_crit=st.number_input("Vibration critical", 1.0, 80.0, 6.0, 0.5),
        att_err_warn=st.number_input("Attitude error warn (°)", 2.0, 60.0, 10.0, 1.0),
        cells=int(st.number_input("Battery cells (0 = auto)", 0, 14, 0)),
        cell_warn=st.number_input("Cell warn (V)", 3.0, 4.0, 3.5, 0.05),
        cell_crit=st.number_input("Cell critical (V)", 2.8, 3.8, 3.3, 0.05),
        capacity_mah=st.number_input("Capacity (mAh, fallback)", 0.0, 50000.0, 0.0, 100.0),
    )

if data is None:
    st.title("UAV Flight Log Analyzer & Telemetry Dashboard")
    st.info("Pick a sample log or upload a PX4 `.ulg` file in the sidebar to begin.")
    st.stop()

log = parse(data, name)
res = analyze(log, cfg)
m = res.metrics
S = log.streams

# ------------------------------------------------------------------ header --
st.title(f"Flight: {log.name}")
c = st.columns(5)
c[0].metric("Health", f"{res.score}/100", grade(res.score))
c[1].metric("Duration", f"{int(log.duration // 60)}:{int(log.duration % 60):02d}")
c[2].metric("Max altitude", f"{m.get('max_alt_m', 0):.0f} m")
c[3].metric("Distance", f"{m['gps']['distance_m']:.0f} m" if m.get("gps") else "n/a")
n_crit = sum(i.severity == "critical" for i in res.issues)
c[4].metric("Issues", len(res.issues), f"{n_crit} critical" if n_crit else "none critical",
            delta_color="inverse" if n_crit else "off")
if any(i.kind == "crash" for i in res.issues):
    st.error("Crash signature: attitude diverged far beyond the commanded values.")

tabs = st.tabs(["Overview", "Attitude", "Battery", "Motors", "Vibration (FFT)", "Issues", "Report"])

# ---------------------------------------------------------------- overview --
with tabs[0]:
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Flight path")
        if log.has("gps"):
            g = S["gps"]
            ok = g[(g["fix"] >= cfg.gps_min_fix) & (g["lat"] != 0)]
            fig = go.Figure()
            fig.add_trace(go.Scattermap(lat=ok["lat"], lon=ok["lon"], mode="lines", name="track",
                                        line=dict(width=4, color="#4cc9f0"),
                                        hovertext=[f"t={t:.0f}s" for t in ok["t"]]))
            lost = g[g["fix"] < cfg.gps_min_fix]
            if len(lost):
                fig.add_trace(go.Scattermap(lat=lost["lat"], lon=lost["lon"], mode="markers", name="GPS lost",
                                            marker=dict(size=11, color="#ef4444")))
            fig.add_trace(go.Scattermap(lat=[ok["lat"].iloc[0]], lon=[ok["lon"].iloc[0]], mode="markers",
                                        name="start", marker=dict(size=13, color="#22c55e")))
            fig.add_trace(go.Scattermap(lat=[ok["lat"].iloc[-1]], lon=[ok["lon"].iloc[-1]], mode="markers",
                                        name="end", marker=dict(size=13, color="#f59e0b")))
            span = max(np.ptp(ok["lat"]), np.ptp(ok["lon"]) * 0.7, 1e-5)
            fig.update_layout(map=dict(style="open-street-map",
                                       center=dict(lat=float(ok["lat"].mean()), lon=float(ok["lon"].mean())),
                                       zoom=float(np.clip(np.log2(360 / span) - 1.2, 1, 19))),
                              height=440, margin=dict(l=0, r=0, t=0, b=0), legend=dict(orientation="h"))
            st.plotly_chart(fig, width="stretch")
        elif log.has("altitude") and "x" in S["altitude"]:
            a = S["altitude"]
            st.plotly_chart(style(go.Figure(go.Scatter(x=a["y"], y=a["x"], mode="lines")), 440), width="stretch")
        else:
            st.info("No GPS or local position data in this log.")
    with right:
        st.subheader("Altitude")
        if log.has("altitude"):
            fig = go.Figure(go.Scatter(x=S["altitude"]["t"], y=S["altitude"]["alt"], name="altitude (m)",
                                       line=dict(color=LINE[0])))
            shade(fig, res.issues)
            st.plotly_chart(style(fig, 215), width="stretch")
        st.subheader("Battery")
        if log.has("battery"):
            b = S["battery"]
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=b["t"], y=b["v"], name="voltage (V)", line=dict(color=LINE[1])))
            fig.add_trace(go.Scatter(x=b["t"], y=b["i"], name="current (A)", line=dict(color=LINE[3])), secondary_y=True)
            shade(fig, res.issues, {"battery"})
            st.plotly_chart(style(fig, 215), width="stretch")
    st.caption("Shaded bands = detected issues (red critical, amber warning).")

# ---------------------------------------------------------------- attitude --
with tabs[1]:
    if log.has("attitude"):
        a = S["attitude"]
        sp = S.get("attitude_sp")
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                            subplot_titles=("Roll (°)", "Pitch (°)", "Yaw (°)", "Tracking error (°)"))
        for r, ax in enumerate(("roll", "pitch", "yaw"), 1):
            fig.add_trace(go.Scatter(x=a["t"], y=a[ax], name=ax, line=dict(color=LINE[r - 1])), row=r, col=1)
            if sp is not None and len(sp) > 1:
                fig.add_trace(go.Scatter(x=sp["t"], y=sp[ax], name=f"{ax} cmd", line=dict(color="#999", dash="dot")), row=r, col=1)
        if "att_err" in res.series:
            e = res.series["att_err"]
            fig.add_trace(go.Scatter(x=e["t"], y=e["err"], name="error", line=dict(color="#ef4444")), row=4, col=1)
            fig.add_hline(y=cfg.att_err_warn, line_dash="dash", line_color="#f59e0b", row=4, col=1)
        for r in range(1, 5):
            shade(fig, res.issues, {"attitude", "crash"}, row=r)
        fig.update_layout(height=760, margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified", legend=dict(orientation="h", y=1.05))
        st.plotly_chart(fig, width="stretch")
        if m.get("attitude"):
            st.write(f"Mean error **{m['attitude']['err_mean']:.1f}°**, 95th percentile **{m['attitude']['err_p95']:.1f}°**, max **{m['attitude']['err_max']:.0f}°**.")
    else:
        st.info("No attitude data.")

# ----------------------------------------------------------------- battery --
with tabs[2]:
    b = m.get("battery")
    if b:
        k = st.columns(5)
        k[0].metric("Pack", f"{b['cells']}S")
        k[1].metric("Voltage", f"{b['v_start']:.1f} → {b['v_end']:.1f} V")
        k[2].metric("Min cell", f"{b['cell_min']:.2f} V")
        k[3].metric("Int. resistance", f"{b['r_int_ohm'] * 1000:.0f} mΩ" if b["r_int_ohm"] else "n/a")
        k[4].metric("Est. time left", f"{b['t_left_s'] / 60:.1f} min" if b["t_left_s"] else "n/a",
                    b["t_left_method"] or None, delta_color="off")
        bd, cv = S["battery"], res.series["cell_v"]
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                            subplot_titles=("Pack voltage (V) & per-cell voltage", "Current (A)", "Sag vs 10 s baseline (V/cell)"))
        fig.add_trace(go.Scatter(x=bd["t"], y=bd["v"], name="pack V", line=dict(color=LINE[1])), row=1, col=1)
        fig.add_trace(go.Scatter(x=cv["t"], y=cv["cell_v"], name="V/cell", line=dict(color=LINE[2])), row=1, col=1)
        fig.add_hline(y=cfg.cell_crit, line_dash="dash", line_color="#ef4444", row=1, col=1)
        fig.add_trace(go.Scatter(x=bd["t"], y=bd["i"], name="current", line=dict(color=LINE[3])), row=2, col=1)
        fig.add_trace(go.Scatter(x=cv["t"], y=cv["sag"], name="sag", line=dict(color=LINE[4])), row=3, col=1)
        fig.add_hline(y=cfg.sag_per_cell, line_dash="dash", line_color="#f59e0b", row=3, col=1)
        for r in (1, 2, 3):
            shade(fig, res.issues, {"battery"}, row=r)
        fig.update_layout(height=720, margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified", legend=dict(orientation="h", y=1.05))
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No battery data.")

# ------------------------------------------------------------------ motors --
with tabs[3]:
    if log.has("motors"):
        mo = S["motors"]
        fig = go.Figure()
        for i, col in enumerate([c for c in mo.columns if c.startswith("m")]):
            fig.add_trace(go.Scatter(x=mo["t"], y=mo[col], name=f"motor {i + 1}", line=dict(color=LINE[i % 6])))
        shade(fig, res.issues, {"crash", "attitude"})
        st.plotly_chart(style(fig, 430, "Motor outputs"), width="stretch")
        if "motor_saturation_pct" in m:
            st.write(f"Time with any motor near max output: **{m['motor_saturation_pct']:.1f}%**")
    else:
        st.info("No motor output data.")

# --------------------------------------------------------------- vibration --
with tabs[4]:
    v = m.get("vibration")
    if v:
        k = st.columns(4)
        k[0].metric("Mean RMS", f"{v['rms_mean']:.2f} m/s²")
        k[1].metric("Max RMS", f"{v['rms_max']:.2f} m/s²")
        k[2].metric("FFT peak", f"{v['peak_freq_hz']:.0f} Hz")
        k[3].metric("IMU rate", f"{v['sample_rate_hz']:.0f} Hz")
        l, r = st.columns(2)
        vr = res.series["vib_rms"]
        fig = go.Figure(go.Scatter(x=vr["t"], y=vr["rms"], name="RMS", line=dict(color=LINE[0])))
        fig.add_hline(y=cfg.vib_warn, line_dash="dash", line_color="#f59e0b")
        fig.add_hline(y=cfg.vib_crit, line_dash="dash", line_color="#ef4444")
        shade(fig, res.issues, {"vibration"})
        l.plotly_chart(style(fig, 380, "Vibration level (1 s RMS, > 5 Hz)"), width="stretch")
        ps = res.series["psd"]
        fig = go.Figure(go.Scatter(x=ps["f"], y=ps["psd"], fill="tozeroy", line=dict(color=LINE[1])))
        fig.update_yaxes(type="log", title_text="PSD")
        fig.update_xaxes(title_text="frequency (Hz)")
        fig.update_layout(height=380, title="Accelerometer spectrum (Welch FFT)", margin=dict(l=10, r=10, t=40, b=10))
        r.plotly_chart(fig, width="stretch")
        st.caption("A sharp peak usually points to a prop/motor imbalance at that frequency. "
                   "Broadband noise points to frame resonance or a poor mount.")
    else:
        st.info("No usable accelerometer data (needs ≥ 50 Hz).")

# ------------------------------------------------------------------ issues --
with tabs[5]:
    if not res.issues:
        st.success("No problems detected.")
    else:
        df = pd.DataFrame([{"severity": f"{ICON[i.severity]} {i.severity}", "check": i.kind,
                            "start (s)": round(i.t_start, 1), "end (s)": round(i.t_end, 1), "detail": i.detail}
                           for i in res.issues])
        st.dataframe(df, width="stretch", hide_index=True)
        # timeline
        fig = go.Figure()
        kinds = list(dict.fromkeys(i.kind for i in res.issues))
        for i in res.issues:
            fig.add_trace(go.Scatter(x=[i.t_start, max(i.t_end, i.t_start + 0.4)], y=[i.kind] * 2, mode="lines",
                                     line=dict(width=14, color="#ef4444" if i.severity == "critical" else "#f59e0b"),
                                     hovertext=i.detail, showlegend=False))
        fig.update_yaxes(categoryorder="array", categoryarray=kinds[::-1])
        st.plotly_chart(style(fig, 70 + 45 * len(kinds), "Issue timeline"), width="stretch")
        for k in kinds:
            st.markdown(f"**{k.capitalize()}** → {ADVICE[k]}")

# ------------------------------------------------------------------ report --
with tabs[6]:
    md = build_markdown(log, res)
    st.markdown(md)
    d1, d2 = st.columns(2)
    d1.download_button("⬇ Markdown", md, f"{log.name}_report.md", "text/markdown")
    d2.download_button("⬇ Printable HTML (Print → PDF)", build_html(log, res), f"{log.name}_report.html", "text/html")
