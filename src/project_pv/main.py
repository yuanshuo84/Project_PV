"""Application entry points for Project PV."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def configure_tcl_paths() -> None:
    """Point Tkinter at Python's bundled Tcl/Tk files when needed."""

    tcl_root = Path(sys.base_prefix) / "tcl"
    tcl_library = tcl_root / "tcl8.6"
    tk_library = tcl_root / "tk8.6"

    if "TCL_LIBRARY" not in os.environ and (tcl_library / "init.tcl").exists():
        os.environ["TCL_LIBRARY"] = str(tcl_library)
        LOGGER.debug("Set TCL_LIBRARY=%s", tcl_library)
    if "TK_LIBRARY" not in os.environ and (tk_library / "tk.tcl").exists():
        os.environ["TK_LIBRARY"] = str(tk_library)
        LOGGER.debug("Set TK_LIBRARY=%s", tk_library)


def configure_logging() -> Path:
    """Configure console and file logging for the desktop app."""

    log_dir = Path.cwd() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "project_pv.log"
    level_name = os.getenv("PROJECT_PV_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    LOGGER.debug("Logging configured at %s", log_path)
    return log_path


def main() -> None:
    """Open the Project PV UI."""

    log_path = configure_logging()
    configure_tcl_paths()
    LOGGER.info("Starting Project PV; log file: %s", log_path)
    run_ui()


def run_ui() -> None:
    """Open the vector-to-GIF animation UI."""

    from project_pv.ui import run_app

    LOGGER.debug("Opening Project PV UI")
    run_app()


if __name__ == "__main__":
    main()
