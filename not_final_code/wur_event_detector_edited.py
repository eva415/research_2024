import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from std_srvs.srv import Empty
from std_msgs.msg import UInt16
from geometry_msgs.msg import WrenchStamped
import numpy as np
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from ur_msgs.msg import IOStates  # Make sure to import the message type
from datetime import datetime

SECONDS_TO_CROP = 30

class EventDetector(Node):

    def __init__(self):

        super().__init__('event_detector')
        self.simulation_mode = True  # Set this to True if running with bag files
        self.cbgroup = ReentrantCallbackGroup()
        if not self.simulation_mode:
            self.stop_controller_cli = self.create_client(Empty, 'stop_controller', callback_group=self.cbgroup)
            self.stop_controller_pull_twist = self.create_client(Empty, 'pull_twist/stop_controller')
            self.wait_for_srv(self.stop_controller_cli)
            self.wait_for_srv(self.stop_controller_pull_twist)
            self.stop_controller_req = Empty.Request()
        else:
            self.start_time = datetime.now()
            print('In simulation mode (replaying bag files)...')

        self.detect_service = self.create_service(Empty, 'detect_events', self.detect_events,
                                                  callback_group=self.cbgroup)
        self.subscriber = self.create_subscription(WrenchStamped, '/ft300_wrench', self.wrench_callback, 10,
                                                   callback_group=self.cbgroup)
        IOStates_msg_type = get_message('ur_msgs/msg/IOStates')
        self.pressure_subscriber = self.create_subscription(IOStates_msg_type, '/io_and_status_controller/io_states', self.pressure_callback, 10,
                                                            callback_group=self.cbgroup)  # WUR to set topic

        self.force_memory = []
        self.pressure_memory = []
        self.window = 10

        self.timer = self.create_timer(0.01, self.timer_callback, callback_group=self.cbgroup)

        self.flag = False
        self.active = 0

        # WUR should adjust these to suit their gripper
        self.engaged_pressure = 550.0
        self.disengaged_pressure = 1000.0
        self.failure_ratio = 0.57

        # don't change this though
        self.pressure_threshold = self.engaged_pressure + self.failure_ratio * (
                    self.disengaged_pressure - self.engaged_pressure)

        # optionally change this (should work well for UR5)
        self.force_change_threshold = -1.0

    def detect_events(self, request, response):

        self.get_logger().info(f'Start event detection!')
        self.active = 1
        return response

    def clear_trial(self):

        self.force_memory = []
        self.pressure_memory = []
        self.flag = False
        self.active = 0

    def stop_controller(self):
        if self.simulation_mode:
            # self.get_logger().info("Simulation mode enabled: Skipping stop_controller logic")
            self.clear_trial()
            return
        self.future = self.stop_controller_cli.call_async(self.stop_controller_req)
        self.future.add_done_callback(self.service_response_callback)
        # rclpy.spin_until_future_complete(self, self.future)
        # self.stop_controller_cli.call(self.stop_controller_req)
        # rclpy.spin_until_future_complete(self, self.future)
        self.future = self.stop_controller_pull_twist.call_async(self.stop_controller_req)

        self.future.add_done_callback(self.service_response_callback)
        # rclpy.spin_until_future_complete(self, self.future_2)

        self.clear_trial()

    def service_response_callback(self, future):
        try:
            response = future.result()
            self.get_logger().info(f'Stopped controller!')
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')

    def wrench_callback(self, msg):
        wrench = msg.wrench
        current_force = np.array([wrench.force.x, wrench.force.y,
                                  wrench.force.z])

        force_mag = np.linalg.norm(current_force)
        # print(f"f: {force_mag}")
        self.force_memory.insert(0, force_mag)
        if len(self.force_memory) > self.window:
            self.force_memory = self.force_memory[0:self.window]

    def pressure_callback(self, msg):
        # expects a scalar as a message
        # msg_type = get_message('ur_msgs/msg/IOStates')
        # pressure = deserialize_message(msg, msg_type)
        pressure = msg
        current_pressure = msg.analog_in_states[1].state
        current_pressure = current_pressure * (-100.) + 1000.
        # print(f"p: {current_pressure}")

        self.pressure_memory.insert(0, current_pressure)
        if len(self.pressure_memory) > self.window:
            self.pressure_memory = self.pressure_memory[0:self.window]

    def moving_average(self, force):

        window_size = 5
        i = 0
        filtered = []

        while i < len(force) - window_size + 1:
            window_average = round(np.sum(force[i:i + window_size]) / window_size, 2)
            filtered.append(window_average)
            i += 1

        return filtered

    def timer_callback(self):
        # print("..............................................")
        elapsed_time = datetime.now() - self.start_time
        elapsed_time_str = str(elapsed_time).split('.')[0]  # Remove microseconds
        if self.simulation_mode:
            self.active = 1
            elapsed_seconds = elapsed_time.total_seconds()
            if elapsed_seconds < SECONDS_TO_CROP:
                return
        # if for some reason the engaged pressure is higher, flip > and < for pressure
        if len(self.force_memory) < self.window or len(self.pressure_memory) < self.window:
            # print("Not enough data to process pick classification yet.")
            return

        if self.active == 0:
            print("System is inactive, skipping pick classification.")
            return

        filtered_force = self.moving_average(self.force_memory)
        avg_pressure = np.average(self.pressure_memory)

        backwards_diff = []
        h = 2
        for j in range(2 * h, (len(filtered_force))):
            diff = ((3 * filtered_force[j]) - (4 * filtered_force[j - h]) + filtered_force[j - (2 * h)]) / (2 * h)
            backwards_diff.append(diff)
            j += 1

        cropped_backward_diff = np.average(np.array(backwards_diff))

        # If the suction cups are disengaged, the pick failed
        if avg_pressure >= self.pressure_threshold:
            print(
                f"Apple was failed to be picked at {elapsed_time_str}! :( Force: {np.round(filtered_force[0])} Max Force: {np.max(filtered_force)}  Bdiff: {cropped_backward_diff}  Pressure: {avg_pressure}")
            self.stop_controller()

        # If there is a reasonable force, but pick hasn't been classified yet
        elif filtered_force[0] >= 5:
            # print("Force threshold met, but pick not yet classified.")
            self.flag = True  # force was achieved

            # Check for a significant force drop
            if float(cropped_backward_diff) <= self.force_change_threshold and avg_pressure < self.pressure_threshold:
                # Print the time when the apple is picked
                print(
                    f"Apple has been picked at {elapsed_time_str}! Bdiff: {cropped_backward_diff}   Pressure: {avg_pressure}.\
                        #Force: {filtered_force[0]} vs. Max Force: {np.max(self.force_memory)}")
                self.stop_controller()

            elif float(
                    cropped_backward_diff) <= self.force_change_threshold and avg_pressure >= self.pressure_threshold:
                print(
                    f"Apple was failed to be picked at {elapsed_time_str}! :( Force: {np.round(filtered_force[0])} Max Force: {np.max(self.force_memory)}  Bdiff: {cropped_backward_diff}  Pressure: {avg_pressure}")
                self.stop_controller()

        # If force is low but was high previously, that's a failure too
        elif self.flag and filtered_force[0] < 4.5:
            print(
                f"Apple was failed to be picked at {elapsed_time_str}! :( Force: {np.round(filtered_force[0])} Max Force: {np.max(self.force_memory)}  Bdiff: {cropped_backward_diff}  Pressure: {avg_pressure}")
            self.stop_controller()

        # If no decision has been made yet, print a message indicating the system is still processing
        # else:
            # print("Pick status is still being processed. Force and/or pressure thresholds not yet met.")

    def wait_for_srv(self, srv):
        while not srv.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f'{srv.srv_name} service not available, waiting again...')


def main():
    rclpy.init()

    node = EventDetector()

    # Create a MultiThreadedExecutor
    executor = MultiThreadedExecutor()

    # Add the node to the executor
    executor.add_node(node)

    try:
        # Spin the executor
        executor.spin()
    finally:
        # Clean up
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()