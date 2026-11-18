"""Create the bundled sample .ulg files:  python scripts/generate_samples.py"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from flightlog.synthetic import make_all  # noqa: E402

if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "..", "samples")
    for p in make_all(out):
        print("wrote", os.path.relpath(p), os.path.getsize(p) // 1024, "KB")
