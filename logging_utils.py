import csv
from datetime import datetime
from pathlib import Path
import os

# We keep logs inside a data/ folder so they are easy to find and git-ignore later if needed.
LOG_PATH = Path("data") / "usage_log.csv"

ANALYTICS_DIR = "analytics"
USAGE_LOG = os.path.join(ANALYTICS_DIR, "usage_log.csv")
FEEDBACK_LOG = os.path.join(ANALYTICS_DIR, "feedback_log.csv")

os.makedirs(ANALYTICS_DIR, exist_ok=True)


def _append_row(path, header, row_dict):
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)


def log_usage(event_type: str, scenario: str | None = None):
    header = ["timestamp", "event_type", "scenario"]
    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "event_type": event_type,
        "scenario": scenario or "",
    }
    _append_row(USAGE_LOG, header, row)


def log_feedback(
    rating: int | None, comment: str, scenario: str | None = None
):
    header = ["timestamp", "rating", "comment", "scenario"]
    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "rating": rating if rating is not None else "",
        "comment": comment,
        "scenario": scenario or "",
    }
    _append_row(FEEDBACK_LOG, header, row)
