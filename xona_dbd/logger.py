from pathlib import Path
import traceback
from datetime import datetime

LOG_DIR = Path(__file__).resolve().parent.parent/"userdata"/"creative"/"XonasDBDChecker"/"logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

def write_log(filename, message):
    with open(LOG_DIR/filename,"a",encoding="utf-8") as f:
        f.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}\n")

def log_exception(filename="ui_crash.log"):
    write_log(filename, traceback.format_exc())
