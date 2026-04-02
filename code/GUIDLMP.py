"""
DLMP GUI

Graphical user interface for the DLMP (Deep Learning Multi-Processing) simulator.

This GUI provides a simplified interface to configure and run distributed
training simulations using:
- SYNC-central communication (mainMASCNN.py)
- P2P-ring communication (mainMASACNN.py)

Supported datasets:
- MNIST
- CIFAR-10
- CIFAR-100
- UA-DETRAC

Author:
Jorge A. Lopez

Affiliation:
Toronto Metropolitan University (TMU)
Department of Computer Science

Project:
DLMP - Deep Learning Multi-Processing Simulator

Repository:
https://github.com/DLMPsim/DLMP

License:
MIT (see LICENSE file)
Copyright (c) Jorge A. Lopez

Note:
This software was developed as part of the author's PhD research at
Toronto Metropolitan University. Copyright is held by the author.
"""

import sys
from pathlib import Path

from PyQt5.QtCore import QProcess, Qt
from PyQt5.QtGui import QFont, QTextCursor
from PyQt5.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class DLMPMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.base_dir = Path(__file__).resolve().parent
        self.sync_script = self.base_dir / "mainMASCNN.py"
        self.p2p_script = self.base_dir / "mainMASACNN.py"

        self.process = QProcess(self)

        self.setWindowTitle("DLMP GUI")
        self.resize(900, 780)

        self.build_ui()
        self.connect_signals()
        self.update_patience_limit()

    def build_ui(self):
        central_widget = QWidget(self)
        central_widget.setStyleSheet("background-color: #f3d1d1;")
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(10)

        title = QLabel("Deep Learning Multi-Processing (DLMP)")
        title_font = QFont()
        title_font.setPointSize(17)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #c70000;")
        main_layout.addWidget(title)

        subtitle = QLabel("Distributed deep learning simulator")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: #7a0000;")
        main_layout.addWidget(subtitle)

        mode_group = QGroupBox("Communication Mode")
        mode_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        mode_layout = QHBoxLayout()

        self.sync_radio = QRadioButton("SYNC-central")
        self.p2p_radio = QRadioButton("P2P-ring")
        self.sync_radio.setChecked(True)

        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.sync_radio)
        self.mode_group.addButton(self.p2p_radio)

        mode_layout.addWidget(self.sync_radio)
        mode_layout.addWidget(self.p2p_radio)
        mode_layout.addStretch()
        mode_group.setLayout(mode_layout)
        main_layout.addWidget(mode_group)

        params_group = QGroupBox("Simulation Parameters")
        params_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(8)

        row = 0

        self.gpu_check = QCheckBox("Use GPU if available")
        grid.addWidget(self.gpu_check, row, 0, 1, 2)
        row += 1

        grid.addWidget(QLabel("Dataset:"), row, 0)
        dataset_layout = QHBoxLayout()
        dataset_layout.setContentsMargins(0, 0, 0, 0)
        dataset_layout.setSpacing(12)

        self.mnist_radio = QRadioButton("MNIST")
        self.cifar10_radio = QRadioButton("CIFAR-10")
        self.cifar100_radio = QRadioButton("CIFAR-100")
        self.uadetrac_radio = QRadioButton("UA-DETRAC")
        self.mnist_radio.setChecked(True)

        self.dataset_group = QButtonGroup(self)
        self.dataset_group.addButton(self.mnist_radio)
        self.dataset_group.addButton(self.cifar10_radio)
        self.dataset_group.addButton(self.cifar100_radio)
        self.dataset_group.addButton(self.uadetrac_radio)

        dataset_layout.addWidget(self.mnist_radio)
        dataset_layout.addWidget(self.cifar10_radio)
        dataset_layout.addWidget(self.cifar100_radio)
        dataset_layout.addWidget(self.uadetrac_radio)
        dataset_layout.addStretch()
        grid.addLayout(dataset_layout, row, 1)
        row += 1

        grid.addWidget(QLabel("Processors:"), row, 0)
        self.processors_spin = QSpinBox()
        self.processors_spin.setRange(1, 512)
        self.processors_spin.setValue(2)
        grid.addWidget(self.processors_spin, row, 1)
        row += 1

        grid.addWidget(QLabel("Batch size:"), row, 0)
        self.batch_combo = QComboBox()
        self.batch_combo.addItems(["16", "32", "64", "128", "256", "512", "1024"])
        self.batch_combo.setCurrentText("64")
        grid.addWidget(self.batch_combo, row, 1)
        row += 1

        grid.addWidget(QLabel("Epochs:"), row, 0)
        self.epochs_combo = QComboBox()
        self.epochs_combo.addItems(["5", "10", "15", "20", "25", "30", "45", "50", "75", "100"])
        self.epochs_combo.setCurrentText("10")
        grid.addWidget(self.epochs_combo, row, 1)
        row += 1

        grid.addWidget(QLabel("Learning rate:"), row, 0)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setDecimals(4)
        self.lr_spin.setRange(0.0001, 1.0)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setValue(0.0100)
        grid.addWidget(self.lr_spin, row, 1)
        row += 1

        grid.addWidget(QLabel("Patience:"), row, 0)
        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(1, 100)
        self.patience_spin.setValue(5)
        grid.addWidget(self.patience_spin, row, 1)
        row += 1

        grid.addWidget(QLabel("Latency X (ms):"), row, 0)
        self.latency_x_spin = QDoubleSpinBox()
        self.latency_x_spin.setDecimals(1)
        self.latency_x_spin.setRange(0.0, 10000.0)
        self.latency_x_spin.setSingleStep(0.5)
        self.latency_x_spin.setValue(1.0)
        grid.addWidget(self.latency_x_spin, row, 1)
        row += 1

        grid.addWidget(QLabel("Latency Y (ms):"), row, 0)
        self.latency_y_spin = QDoubleSpinBox()
        self.latency_y_spin.setDecimals(1)
        self.latency_y_spin.setRange(0.0, 10000.0)
        self.latency_y_spin.setSingleStep(0.5)
        self.latency_y_spin.setValue(10.0)
        grid.addWidget(self.latency_y_spin, row, 1)
        row += 1

        grid.addWidget(QLabel("Capacity max:"), row, 0)
        self.capacity_spin = QDoubleSpinBox()
        self.capacity_spin.setDecimals(2)
        self.capacity_spin.setRange(1.0, 100.0)
        self.capacity_spin.setSingleStep(0.1)
        self.capacity_spin.setValue(2.0)
        grid.addWidget(self.capacity_spin, row, 1)
        row += 1

        grid.addWidget(QLabel("Network bandwidth (Mbps):"), row, 0)
        self.net_bw_spin = QDoubleSpinBox()
        self.net_bw_spin.setDecimals(1)
        self.net_bw_spin.setRange(0.1, 100000.0)
        self.net_bw_spin.setSingleStep(10.0)
        self.net_bw_spin.setValue(100.0)
        grid.addWidget(self.net_bw_spin, row, 1)
        row += 1

        grid.addWidget(QLabel("Minimum GPU memory (GiB):"), row, 0)
        self.min_gpu_mem_spin = QDoubleSpinBox()
        self.min_gpu_mem_spin.setDecimals(1)
        self.min_gpu_mem_spin.setRange(0.0, 128.0)
        self.min_gpu_mem_spin.setSingleStep(0.5)
        self.min_gpu_mem_spin.setValue(2.0)
        grid.addWidget(self.min_gpu_mem_spin, row, 1)
        row += 1

        grid.addWidget(QLabel("Optimizer:"), row, 0)
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["ADAM", "ADAMW", "SGDM", "RMSP"])
        self.optimizer_combo.setCurrentText("SGDM")
        grid.addWidget(self.optimizer_combo, row, 1)
        row += 1

        grid.addWidget(QLabel("Loss:"), row, 0)
        self.loss_combo = QComboBox()
        self.loss_combo.addItems(["CE", "LSCE", "FC", "WCE"])
        self.loss_combo.setCurrentText("CE")
        grid.addWidget(self.loss_combo, row, 1)
        row += 1

        grid.addWidget(QLabel("Activation:"), row, 0)
        self.activation_combo = QComboBox()
        self.activation_combo.addItems(["RELU", "LEAKY_RELU", "ELU", "SELU", "GELU", "MISH"])
        self.activation_combo.setCurrentText("RELU")
        grid.addWidget(self.activation_combo, row, 1)
        row += 1

        params_group.setLayout(grid)
        main_layout.addWidget(params_group)

        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(12)

        self.run_button = QPushButton("RUN SIMULATION")
        self.run_button.setStyleSheet(
            "background-color: #d40000; color: white; font-weight: bold; padding: 8px;"
        )

        self.stop_button = QPushButton("STOP")
        self.stop_button.setStyleSheet(
            "background-color: #8b0000; color: white; font-weight: bold; padding: 8px;"
        )
        self.stop_button.setEnabled(False)

        self.clear_button = QPushButton("CLEAR OUTPUT")
        self.clear_button.setStyleSheet(
            "background-color: #d40000; color: white; font-weight: bold; padding: 8px;"
        )

        self.exit_button = QPushButton("EXIT")
        self.exit_button.setStyleSheet(
            "background-color: #008000; color: white; font-weight: bold; padding: 8px;"
        )

        buttons_layout.addWidget(self.run_button)
        buttons_layout.addWidget(self.stop_button)
        buttons_layout.addWidget(self.clear_button)
        buttons_layout.addWidget(self.exit_button)

        main_layout.addLayout(buttons_layout)

        output_label = QLabel("Output of the program:")
        output_label.setStyleSheet("color: #7a0000; font-weight: bold;")
        main_layout.addWidget(output_label)

        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setStyleSheet("background-color: white;")
        main_layout.addWidget(self.output_text, 1)

        self.processing_label = QLabel("")
        self.processing_label.setStyleSheet("color: orange; font-weight: bold;")
        main_layout.addWidget(self.processing_label)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: green; font-weight: bold;")
        main_layout.addWidget(self.status_label)

    def connect_signals(self):
        self.epochs_combo.currentTextChanged.connect(self.update_patience_limit)

        self.run_button.clicked.connect(self.run_simulation)
        self.stop_button.clicked.connect(self.stop_simulation)
        self.clear_button.clicked.connect(self.clear_output)
        self.exit_button.clicked.connect(self.close)

        self.process.readyReadStandardOutput.connect(self.handle_stdout)
        self.process.readyReadStandardError.connect(self.handle_stderr)
        self.process.finished.connect(self.process_finished)

    def update_patience_limit(self):
        epochs = int(self.epochs_combo.currentText())
        self.patience_spin.setMaximum(max(1, epochs))
        if self.patience_spin.value() > epochs:
            self.patience_spin.setValue(epochs)

    def validate_inputs(self):
        if self.latency_x_spin.value() > self.latency_y_spin.value():
            QMessageBox.warning(
                self,
                "Invalid latency range",
                "Latency X must be less than or equal to Latency Y.",
            )
            return False

        if not self.sync_script.exists():
            QMessageBox.critical(
                self,
                "Missing file",
                f"Could not find:\n{self.sync_script}",
            )
            return False

        if not self.p2p_script.exists():
            QMessageBox.critical(
                self,
                "Missing file",
                f"Could not find:\n{self.p2p_script}",
            )
            return False

        return True

    def selected_script(self):
        return self.sync_script if self.sync_radio.isChecked() else self.p2p_script

    def selected_dataset(self):
        if self.cifar10_radio.isChecked():
            return "CIFAR10"
        if self.cifar100_radio.isChecked():
            return "CIFAR100"
        if self.uadetrac_radio.isChecked():
            return "UA_DETRAC"
        return "MNIST"

    def build_command(self):
        command = [
            sys.executable,
            str(self.selected_script()),
            "--processors", str(self.processors_spin.value()),
            "--batch_size", self.batch_combo.currentText(),
            "--epochs", self.epochs_combo.currentText(),
            "--lr", f"{self.lr_spin.value():.4f}",
            "--patience", str(self.patience_spin.value()),
            "--latency", f"{self.latency_x_spin.value():.1f},{self.latency_y_spin.value():.1f}",
            "--capacity_max", f"{self.capacity_spin.value():.2f}",
            "--net_bw_mbps", f"{self.net_bw_spin.value():.1f}",
            "--min_gpu_mem_gb", f"{self.min_gpu_mem_spin.value():.1f}",
            "--optimizer", self.optimizer_combo.currentText(),
            "--loss", self.loss_combo.currentText(),
            "--activation", self.activation_combo.currentText(),
            "--dataset", self.selected_dataset(),
        ]

        if self.gpu_check.isChecked():
            command.append("--gpu")

        return command

    def run_simulation(self):
        if not self.validate_inputs():
            return

        if self.process.state() != QProcess.NotRunning:
            QMessageBox.information(
                self,
                "Simulation already running",
                "Please stop the current simulation before starting another one.",
            )
            return

        command = self.build_command()
        program = command[0]
        arguments = command[1:]

        self.output_text.clear()
        self.status_label.clear()
        self.processing_label.setText("Processing...")

        self.output_text.append("Launching DLMP simulation...\n")
        self.output_text.append(f"Working directory: {self.base_dir}\n")
        self.output_text.append(f"Command: {' '.join(command)}\n\n")

        self.process.setWorkingDirectory(str(self.base_dir))
        self.process.start(program, arguments)

        if not self.process.waitForStarted(3000):
            self.processing_label.clear()
            QMessageBox.critical(
                self,
                "Launch error",
                "The simulator process could not be started.",
            )
            return

        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def stop_simulation(self):
        if self.process.state() == QProcess.NotRunning:
            return

        self.output_text.append("\nStopping simulation...\n")
        self.processing_label.clear()
        self.status_label.clear()

        self.process.kill()
        self.process.waitForFinished(2000)

    def process_finished(self):
        self.output_text.moveCursor(QTextCursor.End)
        self.output_text.insertPlainText("\nSimulation finished.\n")
        self.output_text.moveCursor(QTextCursor.End)

        self.processing_label.clear()
        self.status_label.setText("Process finished")

        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def handle_stdout(self):
        text = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self.output_text.moveCursor(QTextCursor.End)
        self.output_text.insertPlainText(text)
        self.output_text.moveCursor(QTextCursor.End)

    def handle_stderr(self):
        text = bytes(self.process.readAllStandardError()).decode("utf-8", errors="replace")
        self.output_text.moveCursor(QTextCursor.End)
        self.output_text.insertPlainText(text)
        self.output_text.moveCursor(QTextCursor.End)

    def clear_output(self):
        self.output_text.clear()
        self.processing_label.clear()
        self.status_label.clear()

    def closeEvent(self, event):
        if self.process.state() == QProcess.NotRunning:
            event.accept()
            return

        reply = QMessageBox.question(
            self,
            "Exit DLMP GUI",
            "A simulation is still running. Do you want to stop it and exit?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            self.process.kill()
            self.process.waitForFinished(2000)
            event.accept()
        else:
            event.ignore()


def main():
    app = QApplication(sys.argv)
    window = DLMPMainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
