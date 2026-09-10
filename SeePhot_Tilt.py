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
import statistics
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
    QComboBox,
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
MODE_SINGLE = "single"
MODE_SERIES = "series"
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


@dataclass(frozen=True)
class SeriesSummary:
    seqtilt_percent: float | None
    seqtilt_aberration_fwhm: float | None
    text: str


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


def percentile(values: list[float], fraction: float) -> float:
    """Return a linearly interpolated percentile for a non-empty value list."""

    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def statistic_text(label: str, values: list[float], unit: str, decimals: int) -> str:
    """Format centre, scatter, tail, and range for available numeric values."""

    if not values:
        return f"{label}: n/a"
    scatter = statistics.stdev(values) if len(values) >= 2 else 0.0
    return (
        f"{label}: mean {statistics.mean(values):.{decimals}f}{unit}, "
        f"median {statistics.median(values):.{decimals}f}{unit}, "
        f"σ {scatter:.{decimals}f}{unit}, "
        f"min {min(values):.{decimals}f}{unit}, "
        f"P90 {percentile(values, 0.90):.{decimals}f}{unit}, "
        f"max {max(values):.{decimals}f}{unit} (n={len(values)})"
    )


def numeric_values(results: list[TiltResult], attribute: str) -> list[float]:
    """Return finite numeric values from result attributes, skipping missing headers."""

    values: list[float] = []
    for result in results:
        value = getattr(result, attribute)
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if numeric == numeric and numeric not in {float("inf"), float("-inf")}:
            values.append(numeric)
    return values


class TiltWorker(QThread):
    result_ready = pyqtSignal(object)
    progress = pyqtSignal(int, int)
    series_summary_ready = pyqtSignal(object)
    analysis_done = pyqtSignal(bool, str)

    def __init__(self, source_dir: Path, mode: str):
        super().__init__()
        self.source_dir = source_dir
        self.mode = mode

    def metadata_for(self, path: Path) -> tuple[object | None, object | None]:
        return header_value(path, "FOCUSPOS"), header_value(path, "CCD-TEMP")

    def measure_loaded_image(self, siril: s.SirilInterface) -> tuple[float, float, float]:
        log_before = siril.get_siril_log() or ""
        siril.cmd("tilt")
        measurement = parse_tilt_result((siril.get_siril_log() or "")[len(log_before) :])
        if measurement is None:
            raise RuntimeError("Siril did not return a tilt result; too few usable stars may be present.")
        return measurement

    def make_result(
        self,
        path: Path,
        focus_position: object | None,
        temperature_c: object | None,
        measurement: tuple[float, float, float],
    ) -> TiltResult:
        _tilt_fwhm, tilt_percent, aberration_fwhm = measurement
        return TiltResult(path.name, aberration_fwhm, tilt_percent, focus_position, temperature_c)

    def run_single_mode(self, siril: s.SirilInterface, paths: list[Path]) -> list[TiltResult]:
        results: list[TiltResult] = []
        for index, path in enumerate(paths, start=1):
            focus_position = temperature_c = None
            try:
                focus_position, temperature_c = self.metadata_for(path)
                with tempfile.TemporaryDirectory(prefix="seephot_tilt_") as temporary_name:
                    temporary_dir = Path(temporary_name)
                    (temporary_dir / path.name).symlink_to(path.resolve())
                    siril.cmd("close")
                    siril.cmd(f'cd "{siril_path(temporary_dir)}"')
                    siril.cmd("convert tilt_frame -debayer")
                    siril.cmd('load "tilt_frame_00001.fit"')
                    measurement = self.measure_loaded_image(siril)
                    siril.cmd("close")
                    siril.cmd(f'cd "{siril_path(self.source_dir.parent)}"')
                result = self.make_result(path, focus_position, temperature_c, measurement)
            except Exception as exc:
                result = TiltResult(path.name, None, None, focus_position, temperature_c, str(exc))
            results.append(result)
            self.result_ready.emit(result)
            self.progress.emit(index, len(paths))
        return results

    def series_summary(
        self,
        metadata_results: list[TiltResult],
        seqtilt_measurement: tuple[float, float, float],
    ) -> SeriesSummary:
        _seqtilt_fwhm, seqtilt_percent, seqtilt_aberration = seqtilt_measurement
        text = "\n".join((
            f"Siril seqtilt (entire series): Tilt {seqtilt_percent:.1f} %, Aberration {seqtilt_aberration:.3f} FWHM.",
            "The series mode does not repeat individual tilt measurements.",
            statistic_text("Focus position", numeric_values(metadata_results, "focus_position"), "", 1),
            statistic_text("CCD temperature", numeric_values(metadata_results, "temperature_c"), " °C", 2),
        ))
        return SeriesSummary(seqtilt_percent, seqtilt_aberration, text)

    def run_series_mode(self, siril: s.SirilInterface, paths: list[Path]) -> list[TiltResult]:
        if len(paths) < 2:
            raise ValueError("Series mode needs at least two FITS images.")
        metadata_results: list[TiltResult] = []
        with tempfile.TemporaryDirectory(prefix="seephot_tilt_series_") as temporary_name:
            temporary_dir = Path(temporary_name)
            for path in paths:
                (temporary_dir / path.name).symlink_to(path.resolve())
            siril.cmd("close")
            siril.cmd(f'cd "{siril_path(temporary_dir)}"')
            siril.cmd("convert tilt_frame -debayer")
            log_before = siril.get_siril_log() or ""
            siril.cmd("seqtilt tilt_frame")
            seqtilt_measurement = parse_tilt_result((siril.get_siril_log() or "")[len(log_before) :])
            if seqtilt_measurement is None:
                raise RuntimeError("Siril did not return a seqtilt result for the series.")
            siril.cmd("close")
            for path in paths:
                focus_position = temperature_c = None
                try:
                    focus_position, temperature_c = self.metadata_for(path)
                except Exception as exc:
                    metadata_results.append(TiltResult(path.name, None, None, focus_position, temperature_c, str(exc)))
                else:
                    metadata_results.append(TiltResult(path.name, None, None, focus_position, temperature_c))
            siril.cmd("close")
            siril.cmd(f'cd "{siril_path(self.source_dir.parent)}"')
        _seqtilt_fwhm, seqtilt_percent, seqtilt_aberration = seqtilt_measurement
        series_result = TiltResult(
            f"Entire series ({len(paths)} images)",
            seqtilt_aberration,
            seqtilt_percent,
            None,
            None,
        )
        self.result_ready.emit(series_result)
        self.progress.emit(1, 1)
        self.series_summary_ready.emit(self.series_summary(metadata_results, seqtilt_measurement))
        return [series_result]

    def run(self) -> None:
        siril = s.SirilInterface()
        connected = False
        try:
            paths = direct_fits_paths(self.source_dir)
            if not paths:
                raise FileNotFoundError("No .fit/.fits files were found directly in the selected folder.")
            siril.connect()
            connected = True
            if self.mode == MODE_SERIES:
                self.run_series_mode(siril, paths)
                message = f"Series analysis complete: {len(paths)} images."
            else:
                self.run_single_mode(siril, paths)
                message = f"Single-image analysis complete: {len(paths)} images."
            self.analysis_done.emit(True, message)
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
        self.resize(900, 680)
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
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Single images", MODE_SINGLE)
        self.mode_combo.addItem("Entire series (seqtilt + statistics)", MODE_SERIES)
        self.mode_combo.setToolTip(
            "Single images measures every image independently. Series builds one debayered sequence, "
            "runs Siril seqtilt, and also reports mean, scatter, and range across the individual frames."
        )
        mode_row.addWidget(self.mode_combo)
        mode_row.addStretch(1)
        layout.addLayout(mode_row)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(("Filename", "Aberration [FWHM]", "Tilt [%]", "Focus position", "Temperature [°C]"))
        self.table.setSortingEnabled(True)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)
        self.series_summary_label = QLabel("")
        self.series_summary_label.setWordWrap(True)
        self.series_summary_label.setVisible(False)
        layout.addWidget(self.series_summary_label)
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
        self.series_summary_label.clear()
        self.series_summary_label.setVisible(False)
        self.start_button.setEnabled(False)
        self.browse_button.setEnabled(False)
        self.mode_combo.setEnabled(False)
        self.close_button.setEnabled(False)
        self.status_label.setText("Starting Siril analysis …")
        self.worker = TiltWorker(source_dir, str(self.mode_combo.currentData()))
        self.worker.result_ready.connect(self.add_result)
        self.worker.progress.connect(self.update_progress)
        self.worker.series_summary_ready.connect(self.show_series_summary)
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

    def show_series_summary(self, summary: SeriesSummary) -> None:
        self.series_summary_label.setText(summary.text)
        self.series_summary_label.setVisible(True)

    def analysis_finished(self, success: bool, message: str) -> None:
        self.table.setSortingEnabled(True)
        self.start_button.setEnabled(True)
        self.browse_button.setEnabled(True)
        self.mode_combo.setEnabled(True)
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
