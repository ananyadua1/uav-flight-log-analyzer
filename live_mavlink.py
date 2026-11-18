"""Optional: print live MAVLink telemetry from a simulator (PX4 SITL / ArduPilot SITL).

    python live_mavlink.py udp:127.0.0.1:14550
Requires `pip install pymavlink`.
"""
import sys
from pymavlink import mavutil

conn = mavutil.mavlink_connection(sys.argv[1] if len(sys.argv) > 1 else "udp:127.0.0.1:14550")
conn.wait_heartbeat()
print("connected to system", conn.target_system)
last = {}
while True:
    m = conn.recv_match(type=["ATTITUDE", "GLOBAL_POSITION_INT", "SYS_STATUS", "GPS_RAW_INT"], blocking=True, timeout=5)
    if m is None:
        print("no data for 5 s: link dropout?")
        continue
    last[m.get_type()] = m
    if m.get_type() == "ATTITUDE":
        a, g = last.get("GLOBAL_POSITION_INT"), last.get("SYS_STATUS")
        print(f"roll {m.roll * 57.3:6.1f}  pitch {m.pitch * 57.3:6.1f}  yaw {m.yaw * 57.3:6.1f}"
              + (f"  alt {a.relative_alt / 1000:6.1f} m" if a else "")
              + (f"  batt {g.voltage_battery / 1000:5.2f} V {g.current_battery / 100:5.1f} A" if g else ""))
