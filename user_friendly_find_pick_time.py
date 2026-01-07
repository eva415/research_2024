import math
import os
import csv

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sys
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PyQt6.QtGui import QPixmap, QImage
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import WrenchStamped
from cv_bridge import CvBridge
import rclpy
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import subprocess
from threading import Thread
from datetime import datetime

# ---------------- CONFIG ----------------
BAG_DIR = "/home/imml/Desktop/successful_picks/"
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
CSV_OUT = f"user_pick_times_{RUN_TIMESTAMP}.csv"
# --------------------------------------


class PyQtROS2App(QWidget):
    def __init__(self):
        super().__init__()

        # ROS init
        rclpy.init()
        self.node = Node('pyqt_ros2_subscriber')
        self.bridge = CvBridge()

        # Bag management
        self.bag_files = sorted([
            os.path.join(BAG_DIR, f)
            for f in os.listdir(BAG_DIR)
            if f.endswith(".db3")
        ])
        self.current_bag_index = 0
        self.results = []

        self.bag_process = None

        # ---- STATE FLAG (FIXED POSITION) ----
        self.bag_active = False

        # --- SUBSCRIPTIONS ---
        self.node.create_subscription(
            WrenchStamped,
            '/force_torque_sensor_broadcaster/wrench',
            self.force_callback,
            10
        )
        self.node.create_subscription(Image, '/image_raw', self.image_callback_1, 10)
        self.node.create_subscription(Image, '/camera/mast_camera/color/image_raw', self.image_callback_2, 10)

        # --- DATA ARRAYS ---
        self.times = []
        self.forces = []
        self.max_time_window = 100
        self.time_reference = None  # DO NOT CHANGE

        # --- GUI ---
        main_layout = QHBoxLayout()
        plot_layout = QVBoxLayout()

        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setFixedSize(350, 200)
        plot_layout.addWidget(self.canvas)

        image_layout = QVBoxLayout()
        self.image_label_1 = QLabel("No /image_raw")
        self.image_label_2 = QLabel("No /camera/mast_camera/color/image_raw")
        image_layout.addWidget(self.image_label_1)
        image_layout.addWidget(self.image_label_2)

        self.pick_analysis_time = -10
        self.user_pick_time = -10

        button_layout = QVBoxLayout()
        self.pick_button = QPushButton("Click when pick occurs!")
        self.pick_button.clicked.connect(self.on_pick_clicked)

        self.accept_button = QPushButton("Accept time")
        self.accept_button.clicked.connect(self.accept_time)

        self.retry_button = QPushButton("Retry bag")
        self.retry_button.clicked.connect(self.retry_bag)

        button_layout.addWidget(self.pick_button)
        button_layout.addWidget(self.accept_button)
        button_layout.addWidget(self.retry_button)

        main_layout.addLayout(plot_layout)
        main_layout.addLayout(image_layout)
        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)

        self.start_ros_spin()
        self.load_current_bag()

    # ---------------- BAG CONTROL ----------------
    def load_current_bag(self):
        if self.current_bag_index >= len(self.bag_files):
            self.finish_all_bags()
            return

        self.reset_state()

        bag_path = self.bag_files[self.current_bag_index]
        self.setWindowTitle(os.path.basename(bag_path))

        self.bag_active = False

        self.bag_process = subprocess.Popen([
            'ros2', 'bag', 'play',
            bag_path,
            '--clock'
        ])

        self.bag_active = True

    def reset_state(self):
        self.bag_active = False
        self.times.clear()
        self.forces.clear()
        self.time_reference = None
        self.user_pick_time = -10
        self.pick_analysis_time = -10
        self.pick_button.setText("Click when pick occurs!")
        self.ax.clear()
        self.canvas.draw()

    def accept_time(self):
        if not self.bag_active:
            return
        if self.user_pick_time < 0:
            return

        bag_name = os.path.basename(self.bag_files[self.current_bag_index])
        self.results.append((bag_name, self.user_pick_time))

        if self.bag_process:
            self.bag_process.terminate()

        self.current_bag_index += 1
        self.load_current_bag()

    def retry_bag(self):
        if self.bag_process:
            self.bag_process.terminate()
        self.load_current_bag()

    def finish_all_bags(self):
        self.write_csv()
        self.close()

    # ---------------- BUTTON CLICK ----------------
    def on_pick_clicked(self):
        if self.time_reference is None:
            return
        if self.times:
            self.user_pick_time = round(self.times[-1], 2)
            self.pick_button.setText(f"Time: {self.user_pick_time}")

    # ---------------- FORCE CALLBACK ----------------
    def force_callback(self, msg: WrenchStamped):
        if not self.bag_active:
            return

        ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

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

        while self.times and self.times[0] < rel - self.max_time_window:
            self.times.pop(0)
            self.forces.pop(0)

    # ---------------- IMAGE CALLBACKS ----------------
    def image_callback_1(self, msg):
        self.handle_image(msg, self.image_label_1)

    def image_callback_2(self, msg):
        self.handle_image(msg, self.image_label_2)

    def handle_image(self, msg, label):
        if not self.bag_active:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            h, w, ch = frame.shape
            qt_image = QImage(frame.data, w, h, ch * w, QImage.Format.Format_BGR888)
            label.setPixmap(QPixmap.fromImage(qt_image))
        except Exception:
            return

        if self.time_reference is None:
            return

        ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        rel = ts - self.time_reference
        self.update_plot(rel)

    # ---------------- PLOT ----------------
    def update_plot(self, current_time):
        self.ax.clear()

        if self.times:
            self.ax.plot(self.times, self.forces, label="Force Magnitude (N)")

        if self.user_pick_time > 0:
            self.ax.axvline(self.user_pick_time, color='g', linestyle='--', label="User pick")

        self.ax.set_xlim(
            max(0, current_time - self.max_time_window),
            current_time + 1
        )
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Force magnitude (N)")
        self.ax.legend(loc="upper right")
        self.canvas.draw()

    # ---------------- ROS SPIN ----------------
    def start_ros_spin(self):
        def spin():
            while rclpy.ok():
                rclpy.spin_once(self.node, timeout_sec=0.05)
        Thread(target=spin, daemon=True).start()

    def closeEvent(self, event):
        if self.bag_process:
            self.bag_process.terminate()

        self.write_csv()
        rclpy.shutdown()
        event.accept()

    def write_csv(self):
        if not self.results:
            print("[INFO] No results to save.")
            return

        with open(CSV_OUT, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["bag_file", "user_pick_time"])
            writer.writerows(self.results)

        print(f"[INFO] Saved {len(self.results)} entries to {CSV_OUT}")


def main():
    app = QApplication(sys.argv)
    gui = PyQtROS2App()
    gui.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
