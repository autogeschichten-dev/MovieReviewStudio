"""Entry point for the frozen Windows application."""
from pathlib import Path
import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from movie_review_studio.app import Studio, main


def smoke_startup(marker: Path) -> int:
    """Exercise normal startup and confirm the editor is visible in the event loop."""
    run_event_loop = QApplication.exec

    def checked_event_loop():
        def check_window():
            visible = any(
                isinstance(window, Studio) and window.isVisible()
                for window in QApplication.topLevelWidgets()
            )
            try:
                if visible:
                    marker.write_text("Movie Review Studio ready\n", encoding="utf-8")
            finally:
                QApplication.exit(0 if marker.is_file() and visible else 1)

        QTimer.singleShot(0, check_window)
        return run_event_loop()

    QApplication.exec = staticmethod(checked_event_loop)
    return main()


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--startup-smoke-test":
        marker_path = Path(sys.argv[2]).resolve()
        del sys.argv[1:]
        raise SystemExit(smoke_startup(marker_path))
    raise SystemExit(main())
