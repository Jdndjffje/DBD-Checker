from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


class UIProcess:
    def __init__(self, script_dir):
        self.script_dir = Path(script_dir)
        self.process = None
        self.last_attempt = 0.0

    def ensure_running(self, enabled=True):
        if not enabled:
            return
        if self.process is not None and self.process.poll() is None:
            return

        now = time.monotonic()
        if now - self.last_attempt < 5.0:
            return
        self.last_attempt = now

        launcher = self.script_dir / "XonasDBDCheckerUI.py"
        try:
            flags = (
                getattr(subprocess, "CREATE_NO_WINDOW", 0)
                | getattr(subprocess, "DETACHED_PROCESS", 0)
            )
            self.process = subprocess.Popen(
                [sys.executable, str(launcher), "--standalone-ui"],
                cwd=str(self.script_dir),
                creationflags=flags,
                close_fds=True,
            )

            # Keep the settings UI from competing with the video-processing
            # worker. Failure here is harmless on non-Windows systems.
            try:
                import ctypes
                IDLE_PRIORITY_CLASS = 0x00000040
                ctypes.windll.kernel32.SetPriorityClass(
                    int(self.process._handle),
                    IDLE_PRIORITY_CLASS,
                )
            except Exception:
                pass
            print("[Xona's DBD Checker] Dark tabbed UI started")
        except Exception as exc:
            print(f"[DbdCheck] UI start failed: {exc}")
