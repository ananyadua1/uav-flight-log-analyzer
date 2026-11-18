import os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
from flightlog.synthetic import SCENARIOS, generate, write_ulog
from flightlog.parser import load_ulog
from flightlog.detectors import analyze
from flightlog.report import build_markdown, build_html

EXPECT = {  # scenario -> set of issue kinds that must appear
    "01_normal_flight": set(),
    "02_vibration_issue": {"vibration"},
    "03_battery_sag": {"battery"},
    "04_gps_dropout": {"gps"},
    "05_sensor_dropouts": {"dropout"},
    "06_attitude_tracking": {"attitude"},
    "07_crash_motor_failure": {"crash", "vibration"},  # vibration = impact shock
}

@pytest.fixture(scope="module")
def tmp():
    with tempfile.TemporaryDirectory() as d:
        yield d

@pytest.mark.parametrize("name", list(SCENARIOS))
def test_scenario(name, tmp):
    p = os.path.join(tmp, name + ".ulg")
    write_ulog(p, generate(name, seed=1))
    log = load_ulog(p)
    for s in ("altitude", "gps", "attitude", "battery", "motors", "accel"):
        assert log.has(s), s
    res = analyze(log)
    kinds = {i.kind for i in res.issues if i.severity != "info"}
    assert kinds == EXPECT[name], (name, kinds)
    assert "Flight Health Report" in build_markdown(log, res)
    assert "<table>" in build_html(log, res) or not res.issues
    if name == "01_normal_flight":
        assert res.score == 100
    if name == "07_crash_motor_failure":
        assert res.score < 70

def test_gap_times(tmp):
    p = os.path.join(tmp, "g.ulg")
    write_ulog(p, generate("04_gps_dropout", seed=0))
    gps = [i for i in analyze(load_ulog(p)).issues if i.kind == "gps"]
    assert abs(gps[0].t_start - 45) < 1.0 and abs(gps[0].t_end - 58) < 1.0

def test_vibration_fft_peak(tmp):
    p = os.path.join(tmp, "v.ulg")
    write_ulog(p, generate("02_vibration_issue", seed=0))
    assert abs(analyze(load_ulog(p)).metrics["vibration"]["peak_freq_hz"] - 70) < 1.5
