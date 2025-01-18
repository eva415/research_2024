import math

import matplotlib
matplotlib.use('TkAgg')  # Force Matplotlib to use the TkAgg backend
import matplotlib.pyplot as plt
plt.plot([1, 2, 3], [4, 5, 6])
# plt.show()
import sys
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PyQt6.QtGui import QPixmap, QImage
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge
import cv2
import rclpy
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

FILENAME = "manually doing stuff rn"
PICK_CLASSIFIER_TIME = 44.31

class PyQtROS2App(QWidget):
    def __init__(self):
        super().__init__()

        # Initialize ROS node
        rclpy.init()
        self.node = Node('pyqt_ros2_subscriber')
        self.bridge = CvBridge()

        # ROS Subscriptions
        self.node.create_subscription(TransformStamped, '/tool_pose', self.data_callback, 10)
        self.node.create_subscription(Image, '/camera/camera/color/image_rect_raw', self.image_callback, 10)

        # Initialize GUI components
        self.norm_pos = []
        self.times = []
        self.max_time_window = 50.0

        # Main layout and plot layout
        main_layout = QHBoxLayout()
        plot_layout = QVBoxLayout()

        # Matplotlib plot for random numbers
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)

        # Set the fixed size of the canvas (e.g., width=400, height=300)
        self.canvas.setFixedSize(900, 800)

        plot_layout.addWidget(self.canvas)

        # Image display
        self.image_label = QLabel()
        self.image_label.setText("No image received")
        main_layout.addLayout(plot_layout)
        main_layout.addWidget(self.image_label)

        # Button layout
        self.pick_analysis_time = -10 # initialize pick_analysis_time outside of plot
        self.user_pick_time = -10 # initialize user_pick_time outside of plot
        button_layout = QVBoxLayout()
        self.button = QPushButton("Click when pick occurs!")
        self.button.clicked.connect(self.on_button_clicked)
        button_layout.addWidget(self.button)

        # Add button layout to main layout
        main_layout.addLayout(button_layout)

        # Set up the main window layout
        self.setLayout(main_layout)

        # Start spinning ROS node in the background
        self.start_ros_spin()

    def on_button_clicked(self):
        current_text = self.button.text()
        self.user_pick_time = np.round(self.times[-1],2)
        self.pick_analysis_time = PICK_CLASSIFIER_TIME # initialize pick_analysis time based on result from pick_classifier.py
        # Toggle the label's text
        if current_text == "Click when pick occurs!":
            self.button.setText(f"Time: {self.user_pick_time}")
        else:
            self.button.setText("Click when pick occurs!")

    def data_callback(self, msg: TransformStamped):
        """Update the plot with the new random number and timestamp."""
        timestamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        x_pos = msg.transform.translation.x
        y_pos = msg.transform.translation.y
        norm_position = math.sqrt(x_pos**2 + y_pos**2)
        self.norm_pos.append(norm_position)

        # If this is the first timestamp, use it as the reference
        if len(self.times) == 0:
            self.time_reference = timestamp_sec

        relative_time = timestamp_sec - self.time_reference
        self.times.append(relative_time)

        # Keep only the data within the time window
        while self.times and self.times[0] < relative_time - self.max_time_window:
            self.times.pop(0)
            self.norm_pos.pop(0)
            # self.x_vals.pop(0)
            # self.y_vals.pop(0)

        # Update the plot
        self.ax.clear()
        self.ax.plot(self.times, self.norm_pos, label="NORM tool pose")
        # self.ax.plot(self.times, self.x_vals, label="X tool pose")
        # self.ax.plot(self.times, self.y_vals, label="Y tool pose")
        self.ax.axvline(self.user_pick_time, color='g')
        self.ax.axvline(self.pick_analysis_time, color='r')
        min_x = max(0, relative_time - self.max_time_window)
        max_x = max(self.max_time_window, relative_time + 1)
        self.ax.set_xlim(min_x, max_x)
        # set y axis limits:
        self.ax.set_ylim(0.2, 0.8)
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Tool Pose")
        self.ax.legend(loc='upper right')
        self.canvas.draw()

    def image_callback(self, msg):
        """Update the image display."""
        try:
            image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            h, w, ch = image.shape
            bytes_per_line = ch * w
            qt_image = QImage(image.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)
            self.image_label.setPixmap(QPixmap.fromImage(qt_image))
        except Exception as e:
            self.node.get_logger().error(f"Failed to convert image: {e}")

    def start_ros_spin(self):
        """Start spinning the ROS node in a background thread."""
        from threading import Thread

        def ros_spin():
            while rclpy.ok():
                rclpy.spin_once(self.node, timeout_sec=0.1)

        self.ros_thread = Thread(target=ros_spin, daemon=True)
        self.ros_thread.start()

    def closeEvent(self, event):
        """Clean up when the app is closed."""
        rclpy.shutdown()
        event.accept()


def main():
    app = QApplication(sys.argv)
    pyqt_app = PyQtROS2App()
    pyqt_app.setWindowTitle(FILENAME)
    pyqt_app.resize(800, 600)
    pyqt_app.show()

    try:
        sys.exit(app.exec())
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
