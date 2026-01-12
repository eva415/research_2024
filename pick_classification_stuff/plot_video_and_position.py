import math

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sys
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PyQt6.QtGui import QPixmap, QImage
from rclpy.node import Node
from sensor_msgs.msg import Image
        # ...
from geometry_msgs.msg import TransformStamped, WrenchStamped
from cv_bridge import CvBridge
import cv2
import rclpy
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import subprocess
from threading import Thread

FILENAME = "manually doing stuff rn"
PICK_CLASSIFIER_TIME = 44.31


class PyQtROS2App(QWidget):
    def __init__(self):
        super().__init__()

        # ROS init
        rclpy.init()

        # Launch rosbag
        self.bag_process = subprocess.Popen([
            'ros2', 'bag', 'play',
            '/home/imml/Desktop/successful_picks/final_approach_and_pick_20251030_131808.db3/',
            '--clock'
        ])

        self.node = Node('pyqt_ros2_subscriber')
        self.bridge = CvBridge()

        # --- SUBSCRIPTIONS ---
        self.node.create_subscription(WrenchStamped, '/force_torque_sensor_broadcaster/wrench', self.force_callback, 10)
        self.node.create_subscription(Image, '/image_raw', self.image_callback, 10)
        self.node.create_subscription(TransformStamped, '/tool_pose', self.data_callback, 10)

        # --- DATA ARRAYS ---
        self.times = []
        self.forces = []
        self.max_time_window = 100
        self.time_reference = None  # FIX: only set in force_callback

        # --- GUI ---
        main_layout = QHBoxLayout()
        plot_layout = QVBoxLayout()

        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setFixedSize(900, 800)
        plot_layout.addWidget(self.canvas)

        self.image_label = QLabel("No image received")
        main_layout.addLayout(plot_layout)
        main_layout.addWidget(self.image_label)

        self.pick_analysis_time = -10
        self.user_pick_time = -10
        button_layout = QVBoxLayout()
        self.button = QPushButton("Click when pick occurs!")
        self.button.clicked.connect(self.on_button_clicked)
        button_layout.addWidget(self.button)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

        self.start_ros_spin()

    # -----------------------------
    # BUTTON CLICK
    # -----------------------------
    def on_button_clicked(self):
        if self.times:
            self.user_pick_time = round(self.times[-1], 2)
        self.pick_analysis_time = PICK_CLASSIFIER_TIME

        if self.button.text() == "Click when pick occurs!":
            self.button.setText(f"Time: {self.user_pick_time}")
        else:
            self.button.setText("Click when pick occurs!")

    # -----------------------------
    # FORCE CALLBACK (time zero defined HERE)
    # -----------------------------
    def force_callback(self, msg: WrenchStamped):
        ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # FIX: only force messages define time zero
        if self.time_reference is None:
            self.time_reference = ts
            print(f"[DEBUG] Time zero set from FORCE at {ts}")

        rel = ts - self.time_reference

        fx = msg.wrench.force.x
        fy = msg.wrench.force.y
        fz = msg.wrench.force.z
        f_mag = np.sqrt(fx*fx + fy*fy + fz*fz)

        self.times.append(rel)
        self.forces.append(f_mag)

        # Trim history
        while self.times and self.times[0] < rel - self.max_time_window:
            self.times.pop(0)
            self.forces.pop(0)

    # -----------------------------
    # TOOL POSE CALLBACK
    # -----------------------------
    def data_callback(self, msg):
        if self.time_reference is None:   # FIX: ignore until force arrives
            return

        ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        rel = ts - self.time_reference

        self.times.append(rel)
        self.forces.append(self.forces[-1] if self.forces else 0)

    # -----------------------------
    # IMAGE CALLBACK
    # -----------------------------
    def image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            h, w, ch = frame.shape
            qt_image = QImage(frame.data, w, h, ch * w, QImage.Format.Format_BGR888)
            self.image_label.setPixmap(QPixmap.fromImage(qt_image))
        except Exception as e:
            self.node.get_logger().error(f"Image conversion error: {e}")
            return

        ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # FIX: DO NOT SET TIME ZERO FROM IMAGE
        if self.time_reference is None:
            return  # ignore images until first force message

        rel = ts - self.time_reference

        if len(self.times) == 0:
            self.times.append(rel)
            self.forces.append(0)

        self.update_plot(rel)

    # -----------------------------
    def update_plot(self, current_time):
        self.ax.clear()

        if self.times:
            self.ax.plot(self.times, self.forces, label="Force Magnitude (N)")

        if self.user_pick_time > 0:
            self.ax.axvline(self.user_pick_time, color='g', linestyle='--', label="User pick")
        if self.pick_analysis_time > 0:
            self.ax.axvline(self.pick_analysis_time, color='r', linestyle='--', label="Classifier pick")

        min_x = max(0, current_time - self.max_time_window)
        max_x = current_time + 1
        self.ax.set_xlim(min_x, max_x)

        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Force magnitude (N)")
        self.ax.legend(loc="upper right")

        self.canvas.draw()

    def start_ros_spin(self):
        def spin():
            while rclpy.ok():
                rclpy.spin_once(self.node, timeout_sec=0.05)
        self.ros_thread = Thread(target=spin, daemon=True)
        self.ros_thread.start()

    def closeEvent(self, event):
        if hasattr(self, "bag_process"):
            self.bag_process.terminate()
        rclpy.shutdown()
        event.accept()


def main():
    app = QApplication(sys.argv)
    gui = PyQtROS2App()
    gui.setWindowTitle(FILENAME)
    gui.show()

    try:
        sys.exit(app.exec())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
