"""One-page flight health report (Markdown + printable HTML)."""
from __future__ import annotations

import html
from datetime import datetime

from .detectors import Analysis
from .parser import FlightLog

ADVICE = {
    "vibration": "Balance props, check motor bearings, and soft-mount the flight controller. Re-check the FFT peak after changes.",
    "battery": "Check pack health (internal resistance), use a higher-C battery, or land earlier. Set a conservative low-battery failsafe.",
    "gps": "Check antenna placement/shielding and interference from video/ESC wiring. Verify the failsafe behaviour on GPS loss.",
    "attitude": "Retune rate/attitude PIDs, check motor/ESC response and payload balance. Compare with wind during the segment.",
    "dropout": "Inspect connectors/cables and logging bandwidth (SD card, CPU load). Look for brown-outs.",
    "crash": "Review motor outputs and attitude just before the event: look for a saturated or failed motor/ESC, then fix before flying again.",
}
ICON = {"critical": "🔴", "warning": "🟠", "info": "🔵"}


def grade(score: int) -> str:
    return "GOOD" if score >= 85 else "FAIR" if score >= 60 else "POOR"


def _fmt_time(s):
    return f"{int(s // 60)}:{s % 60:04.1f}"


def build_markdown(log: FlightLog, res: Analysis) -> str:
    m = res.metrics
    L = [f"# Flight Health Report: {log.name}",
         f"_Generated {datetime.now():%Y-%m-%d %H:%M}_", "",
         f"## Verdict: **{grade(res.score)}** (score {res.score}/100)", ""]
    crit = [i for i in res.issues if i.severity == "critical"]
    if any(i.kind == "crash" for i in res.issues):
        L.append("**Crash signature detected**: attitude diverged far beyond commanded values.\n")
    L += ["## Flight summary", "",
          f"- Duration: {_fmt_time(m['duration_s'])} min",
          f"- Max altitude: {m.get('max_alt_m', float('nan')):.1f} m"]
    if m.get("gps"):
        L.append(f"- Ground distance: {m['gps']['distance_m']:.0f} m, satellites {m['gps']['min_sats']}–{m['gps']['max_sats']}")
    b = m.get("battery")
    if b:
        L.append(f"- Battery ({b['cells']}S): {b['v_start']:.2f} V → {b['v_end']:.2f} V, min {b['v_min']:.2f} V, peak {b['i_max']:.0f} A")
        if b["r_int_ohm"]:
            L.append(f"- Estimated pack internal resistance: {b['r_int_ohm'] * 1000:.0f} mΩ")
        if b["t_left_s"] is not None:
            L.append(f"- Estimated flight time remaining at end: {b['t_left_s'] / 60:.1f} min ({b['t_left_method']})")
    v = m.get("vibration")
    if v:
        L.append(f"- Vibration: mean {v['rms_mean']:.1f}, max {v['rms_max']:.1f} m/s² RMS, FFT peak {v['peak_freq_hz']:.0f} Hz")
    a = m.get("attitude")
    if a:
        L.append(f"- Attitude tracking error: mean {a['err_mean']:.1f}°, 95th pct {a['err_p95']:.1f}°, max {a['err_max']:.0f}°")
    L += ["", f"## Findings ({len(res.issues)})", ""]
    if not res.issues:
        L.append("No problems detected.")
    else:
        L += ["| Sev | Check | Time | Detail |", "|---|---|---|---|"]
        for i in res.issues:
            L.append(f"| {ICON[i.severity]} {i.severity} | {i.kind} | {_fmt_time(i.t_start)}–{_fmt_time(i.t_end)} | {i.detail} |")
        L += ["", "## Recommended actions", ""]
        for k in dict.fromkeys(i.kind for i in sorted(res.issues, key=lambda i: i.severity != "critical")):
            L.append(f"- **{k.capitalize()}**: {ADVICE[k]}")
    return "\n".join(L) + "\n"


def build_html(log: FlightLog, res: Analysis) -> str:
    """Printable single page (use the browser's Print → Save as PDF)."""
    import re
    md = build_markdown(log, res)
    out, in_table, in_list = [], False, False
    for line in md.splitlines():
        e = html.escape(line)
        e = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", e)
        e = re.sub(r"_(.+?)_", r"<i>\1</i>", e)
        if line.startswith("|"):
            cells = [c.strip() for c in e.strip("|").split("|")]
            if set(line) <= set("|- "):
                continue
            tag = "th" if not in_table else "td"
            if not in_table:
                out.append("<table>"); in_table = True
            out.append("<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>")
            continue
        if in_table:
            out.append("</table>"); in_table = False
        if line.startswith("- "):
            if not in_list:
                out.append("<ul>"); in_list = True
            out.append(f"<li>{e[2:]}</li>"); continue
        if in_list:
            out.append("</ul>"); in_list = False
        if line.startswith("# "): out.append(f"<h1>{e[2:]}</h1>")
        elif line.startswith("## "): out.append(f"<h2>{e[3:]}</h2>")
        elif line.strip(): out.append(f"<p>{e}</p>")
    if in_table: out.append("</table>")
    if in_list: out.append("</ul>")
    css = ("body{font:12px/1.4 Segoe UI,Arial,sans-serif;max-width:780px;margin:20px auto;color:#111}"
           "h1{font-size:20px;margin:0}h2{font-size:14px;border-bottom:1px solid #999;margin-top:14px}"
           "table{border-collapse:collapse;width:100%}td,th{border:1px solid #bbb;padding:3px 5px;text-align:left;vertical-align:top}"
           "@media print{body{margin:0}}")
    return f"<!doctype html><meta charset=utf-8><title>Flight report</title><style>{css}</style>" + "\n".join(out)
