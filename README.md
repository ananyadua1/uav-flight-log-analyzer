# 🛩️ UAV Flight Log Analyzer & Telemetry Dashboard

Reads PX4 `.ulg` flight logs, plots what happened, **finds problems automatically** and writes a one-page flight-health report: the same loop UAV engineers use to improve the next iteration.

**Stack:** Python · pyulog · Pandas · NumPy · SciPy · Streamlit · Plotly (+ optional pymavlink, embedded C on Wokwi)

## Features
| | |
|---|---|
| Extraction | altitude, GPS track, roll/pitch/yaw (+ commanded), battery V/I, motor outputs, accelerometer |
| Dashboard | flight path on a map, altitude & battery vs time, attitude vs command, motors, FFT spectrum, issue timeline |
| Vibration | accel high-passed (>5 Hz), 1 s RMS vs limits, **Welch FFT** to find the dominant frequency |
| Battery sag | V-vs-I regression → internal resistance, sag vs 10 s baseline, low-cell checks, **flight-time-left estimate** |
| GPS dropouts | fix_type < 3 intervals and missing-message gaps |
| Attitude tracking | quaternion error between actual and commanded attitude; > 45° flagged as **crash signature** |
| Sensor dropouts | gaps in any stream > 5× its normal period |
| Report | health score, findings table, recommended actions; download as Markdown or printable HTML |

## Quick start
```bash
pip install -r requirements.txt
python scripts/generate_samples.py     # (re)creates samples/*.ulg, already included
streamlit run app.py
python -m pytest tests                 # 9 tests
```
Upload your own log in the sidebar, or pick one of the 7 bundled samples.

## Using real logs (step 1 of the plan)
1. Go to <https://review.px4.io> → *Browse*, open a flight → *Download* the `.ulg`. Grab 5-10; search for ones with a crash/failsafe.
2. Put them in `samples/` (they appear in the picker) or upload them in the app.
3. Tune thresholds in the sidebar (cell count, vibration limits...) — real airframes differ.

> The bundled samples are **synthetic but written in the real ULog binary format** (`flightlog/synthetic.py`) with known injected faults (vibration at 70 Hz, GPS loss 45-58 s, motor failure at 70 s, ...), so the detectors are unit-tested against ground truth. They are not real flights. Replace them with real logs for your portfolio screenshots.

## Deploy on Streamlit Cloud
1. Push this folder to a GitHub repo (keep `samples/`, ~14 MB).
2. <https://share.streamlit.io> → *New app* → pick the repo, main file `app.py`.
3. Add the live link to your resume.

## Project layout
```
app.py                  Streamlit dashboard
flightlog/parser.py     pyulog -> DataFrames (topic fallbacks for PX4 versions)
flightlog/detectors.py  the 5 problem checks + health score
flightlog/report.py     Markdown / HTML report
flightlog/synthetic.py  ULog writer + fault-injected flight generator
live_mavlink.py         optional live telemetry from a simulator
firmware/wokwi_mpu6050_uart/   embedded C demo (I2C + UART), see its README
tests/                  pytest
```

## Known limits / ideas
- Thresholds are sensible defaults, not certified limits; tune per vehicle.
- Real PX4 versions rename some topics (`sensor_gps`, `actuator_motors`...): fallbacks exist for the common ones; extend `parser.py` if a field is missing.
- Next: compare two flights, add EKF innovation / `vehicle_status` mode timeline, PDF export.
