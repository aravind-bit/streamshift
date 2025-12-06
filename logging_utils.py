import csv
from datetime import datetime
from pathlib import Path

# We keep logs inside a data/ folder so they are easy to find and git-ignore later if needed.
LOG_PATH = Path("data") / "usage_log.csv"


def log_usage(buyer: str, target: str, scenario: str | None = None) -> None:
    """
    Append a simple usage record to data/usage_log.csv.
    Does NOT log any personal / identifying info.
    Safe to call on each scenario run.
    """
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        is_new = not LOG_PATH.exists()

        with LOG_PATH.open("a", newline="") as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow(["timestamp_utc", "buyer", "target", "scenario"])
            writer.writerow([
                datetime.utcnow().isoformat(),
                buyer,
                target,
                scenario or ""
            ])
    except Exception as e:
        # We don't want logging errors to ever break the app,
        # so we swallow exceptions here.
        print(f"[usage logging error] {e}")