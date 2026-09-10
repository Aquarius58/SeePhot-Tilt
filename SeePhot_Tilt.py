"""SeePhot Tilt

Read-only per-frame Siril tilt analysis for original Seestar FITS images.

Run this script from Siril's Python script menu, select a directory containing
the individual ``.fit``/``.fits`` frames, then click Start.  For every image
Siril's ``tilt`` command supplies the sensor-tilt and off-axis-aberration
measurements.  ``FOCUSPOS`` and ``CCD-TEMP`` are read unchanged from the FITS
primary header.  The selected directory and its files are never modified.
"""

from __future__ import annotations

import importlib.util
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import sirilpy as s

if not s.utility.check_module_version(">=1.0.13"):
    print("Error: sirilpy module is too old and does not support this script.")
    sys.exit(1)


def ensure_importable_module(import_name: str, package_name: str | None = None) -> None:
    """Install a required package through Siril when it is unavailable."""

    if importlib.util.find_spec(import_name) is None:
        s.ensure_installed(package_name or import_name)


ensure_importable_module("PyQt6")
ensure_importable_module("astropy")

from astropy.io import fits  # noqa: E402
from PyQt6.QtCore import QThread, pyqtSignal  # noqa: E402
from PyQt6.QtWidgets import (  # noqa: E402
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


SCRIPT_VERSION = "0.1.0"
WINDOW_TITLE = f"SeePhot Tilt {SCRIPT_VERSION}"
FITS_SUFFIXES = {".fit", ".fits"}
TILT_RESULT_PATTERN = re.compile(
    r"(?:Sensor tilt|Sensorverkippung)\[FWHM\]:\s*([0-9]+(?:[.,][0-9]+)?)\s*"
    r"\(([0-9]+(?:[.,][0-9]+)?)%\),\s*"
    r"(?:Off-axis aberration|Achsabweichung)\[FWHM\]:\s*([+-]?[0-9]+(?:[.,][0-9]+)?)"
)

APP_DARK_STYLESHEET = """
QWidget { background: #202124; color: #eceff4; }
QLineEdit, QTableWidget { background: #17191d; border: 1px solid #3c4048; color: #eceff4; }
QHeaderView::section { background: #2b3038; color: #eceff4; padding: 5px; border: 1px solid #3c4048; }
QPushButton { background: #2f6fed; color: white; border: 0; border-radius: 4px; padding: 6px 10px; }
QPushButton:disabled { background: #465064; color: #b8c0cc; }
"""


@dataclass(frozen=True)
class TiltResult:
    filename: str
    aberration_fwhm: float | None
    tilt_percent: float | None
    focus_position: object | None
    temperature_c: object | None
    note: str = ""


def direct_fits_paths(source_dir: Path) -> list[Path]:
    """Return only FITS files directly contained in the selected directory."""

    return sorted(
        path for path in source_dir.iterdir() if path.is_file() and path.suffix.lower() in FITS_SUFFIXES
    )


def siril_path(path: Path) -> str:
    """Return a path safe for a quoted Siril command."""

    return str(path.resolve()).replace('"', '\\"')


def header_value(path: Path, keyword: str) -> object | None:
    """Read one primary-header value without changing the FITS file."""

    with fits.open(path, memmap=False) as hdul:
        return hdul[0].header.get(keyword)


def parse_tilt_result(siril_log: str) -> tuple[float, float, float] | None:
    """Extract Siril's tilt FWHM, tilt percentage, and aberration from its command log."""

    matches = list(TILT_RESULT_PATTERN.finditer(siril_log))
    if not matches:
        return None
    tilt_text, percent_text, aberration_text = matches[-1].groups()
    return (
        float(tilt_text.replace(",", ".")),
        float(percent_text.replace(",", ".")),
        float(aberration_text.replace(",", ".")),
    )


class TiltWorker(QThread):
    result_ready = pyqtSignal(object)
    progress = pyqtSignal(int, int)
    analysis_done = pyqtSignal(bool, str)

    def __init__(self, source_dir: Path):
        super().__init__()
        self.source_dir = source_dir

    def run(self) -> None:
        siril = s.SirilInterface()
        connected = False
        try:
            paths = direct_fits_paths(self.source_dir)
            if not paths:
                raise FileNotFoundError("No .fit/.fits files were found directly in the selected folder.")
            siril.connect()
            connected = True
            for index, path in enumerate(paths, start=1):
                focus_position = temperature_c = None
                try:
                    focus_position = header_value(path, "FOCUSPOS")
                    temperature_c = header_value(path, "CCD-TEMP")
                    # Siril requires at least two images for ``seqtilt``.  Build
                    # one debayered image in an automatically removed directory
                    # and apply its identical single-image ``tilt`` measurement.
                    # The source FITS itself is only linked and never written.
                    with tempfile.TemporaryDirectory(prefix="seephot_tilt_") as temporary_name:
                        temporary_dir = Path(temporary_name)
                        input_link = temporary_dir / path.name
                        input_link.symlink_to(path.resolve())
                        siril.cmd("close")
                        siril.cmd(f'cd "{siril_path(temporary_dir)}"')
                        siril.cmd("convert tilt_frame -debayer")
                        siril.cmd('load "tilt_frame_00001.fit"')
                        log_before = siril.get_siril_log() or ""
                        siril.cmd("tilt")
                        measurement = parse_tilt_result((siril.get_siril_log() or "")[len(log_before) :])
                        siril.cmd("close")
                        siril.cmd(f'cd "{siril_path(self.source_dir.parent)}"')
                    if measurement is None:
                        raise RuntimeError("Siril did not return a tilt result; too few usable stars may be present.")
                    _tilt_fwhm, tilt_percent, aberration_fwhm = measurement
                    result = TiltResult(path.name, aberration_fwhm, tilt_percent, focus_position, temperature_c)
                except Exception as exc:
                    result = TiltResult(path.name, None, None, focus_position, temperature_c, str(exc))
                self.result_ready.emit(result)
                self.progress.emit(index, len(paths))
            self.analysis_done.emit(True, f"Analysis complete: {len(paths)} images.")
        except Exception as exc:
            self.analysis_done.emit(False, str(exc))
        finally:
            if connected:
                try:
                    siril.cmd("close")
                    siril.disconnect()
                except Exception:
                    pass


class TiltWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.worker: TiltWorker | None = None
        self.source_dir = Path.home()
        self.init_ui()

    def init_ui(self) -> None:
        self.setWindowTitle(WINDOW_TITLE)
        self.setStyleSheet(APP_DARK_STYLESHEET)
        self.resize(900, 580)
        layout = QVBoxLayout(self)
        note = QLabel(
            "Select a folder with individual FITS frames. Siril measures Tilt and off-axis Aberration "
            "for every image; FOCUSPOS and CCD-TEMP are read from its FITS header."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        source_row = QHBoxLayout()
        self.source_edit = QLineEdit(str(self.source_dir))
        self.browse_button = QPushButton("Choose Folder")
        self.browse_button.clicked.connect(self.choose_directory)
        source_row.addWidget(self.source_edit)
        source_row.addWidget(self.browse_button)
        layout.addLayout(source_row)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(("Filename", "Aberration [FWHM]", "Tilt [%]", "Focus position", "Temperature [°C]"))
        self.table.setSortingEnabled(True)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)
        button_row = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_analysis)
        self.status_label = QLabel("Ready")
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close)
        button_row.addWidget(self.start_button)
        button_row.addWidget(self.status_label)
        button_row.addStretch(1)
        button_row.addWidget(self.close_button)
        layout.addLayout(button_row)

    def choose_directory(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "Select Folder with FITS Frames", self.source_edit.text())
        if selected:
            self.source_edit.setText(selected)

    def start_analysis(self) -> None:
        source_dir = Path(self.source_edit.text().strip()).expanduser()
        if not source_dir.is_dir():
            QMessageBox.warning(self, WINDOW_TITLE, "Select an existing folder.")
            return
        self.table.setSortingEnabled(False)
        self.table.setRowCount(0)
        self.start_button.setEnabled(False)
        self.browse_button.setEnabled(False)
        self.close_button.setEnabled(False)
        self.status_label.setText("Starting Siril analysis …")
        self.worker = TiltWorker(source_dir)
        self.worker.result_ready.connect(self.add_result)
        self.worker.progress.connect(self.update_progress)
        self.worker.analysis_done.connect(self.analysis_finished)
        self.worker.start()

    def add_result(self, result: TiltResult) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        values = (
            result.filename,
            "—" if result.aberration_fwhm is None else f"{result.aberration_fwhm:.3f}",
            "—" if result.tilt_percent is None else f"{result.tilt_percent:.1f} %",
            "—" if result.focus_position is None else str(result.focus_position),
            "—" if result.temperature_c is None else str(result.temperature_c),
        )
        for column, value in enumerate(values):
            item = QTableWidgetItem(value)
            if column == 0 and result.note:
                item.setToolTip(result.note)
            self.table.setItem(row, column, item)

    def update_progress(self, current: int, total: int) -> None:
        self.status_label.setText(f"Analyzing image {current} of {total} …")

    def analysis_finished(self, success: bool, message: str) -> None:
        self.table.setSortingEnabled(True)
        self.start_button.setEnabled(True)
        self.browse_button.setEnabled(True)
        self.close_button.setEnabled(True)
        self.status_label.setText(message)
        if not success:
            QMessageBox.critical(self, WINDOW_TITLE, message)


def run_app() -> None:
    app = QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QApplication(sys.argv)
    window = TiltWindow()
    window.show()
    if owns_app:
        app.exec()


if __name__ == "__main__":
    run_app()
