"""Chạy bộ kiểm thử tự động của envi_lib ngoài Colab (CPU là đủ, ≈1–2 phút).

Cách dùng (từ thư mục gốc của repo):
    pip install -r requirements.txt
    python tests/run_selftest.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from envi_lib import selftest  # noqa: E402

if __name__ == "__main__":
    sys.exit(0 if selftest.run_all() else 1)
