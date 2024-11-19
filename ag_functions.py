import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
from scipy.ndimage import median_filter
from scipy.signal import butter, filtfilt
import rosbag
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

def filter_force(variables, param):
    # Median Filter
    filtered = []
    for i in range(len(variables)):
        temp = median_filter(variables[i], param)
        filtered.append(temp)

    return filtered

def moving_average(final_force):
    window_size = 5
    i = 0
    filtered = []

    while i < len(final_force) - window_size + 1:
        window_average = round(np.sum(final_force[i:i + window_size]) / window_size, 2)
        filtered.append(window_average)
        i += 1

    return filtered

# Median filter
def filter_force_m(variables, param):
    filtered = []
    for i in range(len(variables)):
        temp = median_filter(variables[i], param)
        filtered.append(temp)

    return filtered

# Calculate the elapsed time relative to the first timestamp in an array
def elapsed_time(time_array):
    # Convert to a NumPy array if it's not already (in case a list is passed)
    time_array = np.array(time_array)

    # Check if time_array is not empty
    if time_array.size == 0:
        print("Error: time_array is empty.")
        return None

    # Subtract the first entry from each element (vectorized operation)
    elapsed_t = time_array - time_array[0]

    # Optionally print for debugging
    print(f"elapsed_time: {elapsed_t}")

    return elapsed_t
# Combine seconds and nanoseconds arrays into a single timestamp in seconds
def total_time(seconds, nseconds):
    # Convert seconds and nanoseconds to total time in seconds using vectorized operations
    seconds = np.array(seconds, dtype=np.float64)
    nseconds = np.array(nseconds, dtype=np.float64)

    # Convert nanoseconds to seconds and add to seconds
    total = seconds + (nseconds / 1e9)

    # Optionally print for debugging
    print(f"total_time: {total}")

    return total

# Convert ROS bag file data to CSV format with specific topic data extraction
def bag_to_csv(i):
    file = str(i)
    bag = rosbag.Bag('./' + file + '.bag')
    topic = 'wrench'
    df = []

    # Extract force data from 'wrench' topic in bag file
    for topic, msg, t in bag.read_messages(topics=topic):
        Fx = msg.wrench.force.x
        Fy = msg.wrench.force.y
        Fz = msg.wrench.force.z
        nsecs = msg.header.stamp.nsecs
        secs = msg.header.stamp.secs
        new_values = [Fx, Fy, Fz, nsecs, secs]
        df.append(new_values)  # Append extracted values to data list

    np.savetxt(file + '.csv', df, delimiter=",")  # Save data to CSV file
    return file

# Apply a low-pass Butterworth filter to a data array
def butter_lowpass_filter(data, cutoff, fs, order):
    nyq = 0.5 * fs  # Calculate Nyquist Frequency
    normal_cutoff = cutoff / nyq  # Normalize cutoff frequency
    # Get filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data, padlen=0)  # Apply the filter
    return y

# Match timestamps and interpolate data arrays based on a common timestamp array
def match_times(t1, t2, x1, x2):
    all_timestamps = sorted(set(t1).union(set(t2)))  # Combine and sort unique timestamps
    new_x1 = np.zeros([len(all_timestamps), x1.shape[1]])
    new_x2 = np.zeros([len(all_timestamps), x2.shape[1]])

    # Interpolate each column of x1 and x2 data to match common timestamps
    x1_num_cols = np.size(x1, 1)
    for i in range(x1_num_cols):
        new_x1[:, i] = np.interp(all_timestamps, t1, x1[:, i])

    x2_num_cols = np.size(x2, 1)
    for i in range(x2_num_cols):
        new_x2[:, i] = np.interp(all_timestamps, t2, x2[:, i])

    return new_x1, new_x2, all_timestamps

# Compute the absolute difference of each element from the first element in a list of arrays
def flipping_data(cropped_x):
    new_x = []
    for i in range(len(cropped_x)):
        x = np.absolute(cropped_x[i] - cropped_x[0])  # Calculate difference from first element
        new_x.append(x)
    return new_x

# Extract pressure data from specific topics in a ROS bag file and store it in an array
def bag_pressure(file):
    pressure1 = []
    pressure2 = []
    pressure3 = []
    pressure = []

    bag = rosbag.Bag('./' + file + '.bag')
    topic1 = '/gripper/pressure/sc1'
    topic2 = '/gripper/pressure/sc2'
    topic3 = '/gripper/pressure/sc3'

    # Read pressure data from each topic and store in lists
    for topic1, msg, t in bag.read_messages(topics=topic1):
        P1 = msg.data
        pressure1.append(P1)

    for topic2, msg, t in bag.read_messages(topics=topic2):
        P2 = msg.data
        pressure2.append(P2)

    for topic3, msg, t in bag.read_messages(topics=topic3):
        P3 = msg.data
        pressure3.append(P3)

    # Combine pressure data from all topics into a single array
    for i in range(len(pressure1)):
        new_pressure = [pressure1[i], pressure2[i], pressure3[i]]
        pressure.append(new_pressure)

    pressure_array = np.array(pressure)
    return pressure_array

# Store seconds and nanoseconds for each pressure sensor topic
def pressure_time(file):
    pt1s = []  # Seconds for sensor 1
    pt1n = []  # Nanoseconds for sensor 1
    pt2s = []  # Seconds for sensor 2
    pt2n = []  # Nanoseconds for sensor 2
    pt3s = []  # Seconds for sensor 3
    pt3n = []  # Nanoseconds for sensor 3
    pts = []   # Averaged seconds across sensors
    ptn = []   # Averaged nanoseconds across sensors

    # Load the .bag file for reading
    bag = rosbag.Bag('./' + file + '.bag')

    # Define topics for each pressure sensor
    topic1 = '/gripper/pressure/sc1'
    topic2 = '/gripper/pressure/sc2'
    topic3 = '/gripper/pressure/sc3'

    # Extract timestamp for each message in topic1 and append to sensor 1 lists
    for topic1, msg, t in bag.read_messages(topics=topic1):
        t1s = t.secs
        t1n = t.nsecs
        pt1s.append(t1s)
        pt1n.append(t1n)

    # Extract timestamp for each message in topic2 and append to sensor 2 lists
    for topic2, msg, t in bag.read_messages(topics=topic2):
        t2s = t.secs
        t2n = t.nsecs
        pt2s.append(t2s)
        pt2n.append(t2n)

    # Extract timestamp for each message in topic3 and append to sensor 3 lists
    for topic3, msg, t in bag.read_messages(topics=topic3):
        t3s = t.secs
        t3n = t.nsecs
        pt3s.append(t3s)
        pt3n.append(t3n)

    # Calculate the average seconds across all three sensors for each timestamp
    for i in range(len(pt1s)):
        new = int(pt1s[i] + pt2s[i] + pt3s[i]) / 3
        pts.append(new)

    # Calculate the average nanoseconds across all three sensors for each timestamp
    for i in range(len(pt1n)):
        new = int(pt1n[i] + pt2n[i] + pt3n[i]) / 3
        ptn.append(new)

    # Combine the averaged seconds and nanoseconds into a total time array
    times = total_time(pts, ptn)

    # Compute the elapsed time from the start for each timestamp in `times`
    etimes_pressure = elapsed_time(times, times)

    return etimes_pressure

# Reads a .db3 ros2 bag file and extracts relevant force data to a csv file
# NOT IN PICK_CLASSIFIER.PY
def db3_to_csv_f(folder_name):
    # Convert folder_name to a string to use as the base file name
    name = str(folder_name)
    # print(name)
    # Initialize an empty list to store data from each message
    df = []

    # Pass the folder name containing the metadata.yaml file to Reader
    with Reader(name) as reader:
        # Iterate over each message in the bag file
        for connection, timestamp, rawdata in reader.messages():
            # Check if the message is from the wrench topic (force-torque sensor)
            if connection.topic == '/ft300_wrench':
                # Deserialize the message data using CDR (Common Data Representation) format
                msg = deserialize_cdr(rawdata, connection.msgtype)

                # Extract force components along x, y, and z axes
                Fx = msg.wrench.force.x
                Fy = msg.wrench.force.y
                Fz = msg.wrench.force.z

                # Extract timestamp (seconds and nanoseconds) from message header
                secs = msg.header.stamp.sec
                nsecs = msg.header.stamp.nanosec

                # Compile extracted data into a list
                new_values = [Fx, Fy, Fz, secs, nsecs]
                # Append this list to the main data list
                df.append(new_values)

    # Check if data was collected before attempting to save to CSV
    if df:
        # Save the accumulated data to a CSV file using the provided folder_name as the file name
        # print(df)
        np.savetxt(name + 'force.csv', np.array(df), delimiter=",")
    else:
        print("No data found in the specified topic.")

    # Return the folder name as a confirmation of successful save
    return name + 'force'


# Reads a .db3 ros2 bag file and extracts relevant pressure data to a csv file
# NOT IN PICK_CLASSIFIER.PY
import matplotlib.pyplot as plt
def db3_to_csv_p(folder_name):
    # Convert folder_name to a string to use as the base file name
    name = str(folder_name)
    # print(name)
    # Initialize an empty list to store data from each message
    df = []

    # Pass the folder name containing the metadata.yaml file to Reader
    with Reader(name) as reader:
        # Iterate over each message in the bag file
        for connection, timestamp, rawdata in reader.messages():
            # Check if the message is from the wrench topic (force-torque sensor)
            if connection.topic == '/io_and_status_controller/io_states':
                # Deserialize the message data using CDR (Common Data Representation) format
                # msg = deserialize_cdr(rawdata, connection.msgtype)
                # print(f"RAW DATA: {rawdata}")
                msg_type = get_message('ur_msgs/msg/IOStates')
                msg = deserialize_message(rawdata, msg_type)
                # print(f"MSG: {msg}")
                # Extract pressure components
                P0 = msg.analog_in_states[0].state
                # print(f"P0: {P0}")
                P1 = msg.analog_in_states[1].state
                # print(f"P1: {P1}")


                # Extract timestamp (seconds and nanoseconds) from message header
                secs = timestamp // 1_000_000_000  # Get the seconds part
                nsecs = timestamp % 1_000_000_000  # Get the nanoseconds part
                # print(f"secs: {secs}")
                # print(f"nsecs: {nsecs}\n")
                #
                # Compile extracted data into a list
                new_values = [P0, P1, secs, nsecs]
                # Append this list to the main data list
                df.append(new_values)

    # Check if data was collected before attempting to save to CSV
    if df:
        # print(f"Dimensions of df: {len(df)} rows, {len(df[0])} columns")
        # Save the accumulated data to a CSV file using the provided folder_name as the file name
        np.savetxt(name + 'pressure.csv', np.array(df), delimiter=",")
        # plt.figure(figsize=(10, 6))
        # # plt.plot([row[2] for row in df], label='Seconds', color='blue', alpha=0.7)
        # plt.plot([row[0] for row in df], label='Pressure0', color='red', alpha=0.7)
        # plt.plot([row[1] for row in df], label='Pressure1', color='blue', alpha=0.7)
        # plt.plot([row[2] for row in df], label='secs', color='green', alpha=0.7)
        # plt.plot([row[3] for row in df], label='nsecs', color='black', alpha=0.7)
        # plt.xlabel('Index')
        # plt.ylabel('Time')
        # plt.title('Time Series: Seconds and Nanoseconds')
        # plt.legend()
        # plt.grid(True)
        # plt.tight_layout()
        #
        # # Display the plot instead of saving it
        # plt.show()  # This will pop up the plot in a window
    else:
        print("No data found in the specified topic.")

    # Return the folder name as a confirmation of successful save
    return name + 'pressure'


# Reads a .db3 ros2 bag file and extracts relevant position data to a csv file
# NOT IN PICK_CLASSIFIER.PY
def db3_to_csv_x(folder_name):
    # Convert folder_name to a string to use as the base file name
    name = str(folder_name)
    # print(name)
    # Initialize an empty list to store data from each message
    df = []

    # Pass the folder name containing the metadata.yaml file to Reader
    with Reader(name) as reader:
        # Iterate over each message in the bag file
        for connection, timestamp, rawdata in reader.messages():
            # Check if the message is from the wrench topic (force-torque sensor)
            if connection.topic == '/tool_pose':
                # Deserialize the message data using CDR (Common Data Representation) format
                msg = deserialize_cdr(rawdata, connection.msgtype)
                # Extract force components along x, y, and z axes
                x_pos = msg.transform.translation.x
                y_pos = msg.transform.translation.y
                z_pos = msg.transform.translation.z

                # Extract timestamp (seconds and nanoseconds) from message header
                secs = msg.header.stamp.sec
                nsecs = msg.header.stamp.nanosec

                # Compile extracted data into a list
                new_values = [x_pos, y_pos, z_pos, secs, nsecs]
                # Append this list to the main data list
                df.append(new_values)
                # print(new_values)

    # Check if data was collected before attempting to save to CSV
    if df:
        # Save the accumulated data to a CSV file using the provided folder_name as the file name
        # print(df)
        np.savetxt(name + 'pos.csv', np.array(df), delimiter=",")
    else:
        print("No data found in the specified topic.")

    # Return the folder name as a confirmation of successful save
    return name + 'pos'


# Applies a median filter to each variable --> noise reduction
def filter_force(variables, param):
    # Median Filter
    filtered = []
    for i in range(len(variables)):
        temp = median_filter(variables[i], param)
        filtered.append(temp)

    return filtered

# Calculates the moving average of the input data over a defined window size --> smoothing data
def moving_average(final_force):
    window_size = 5
    i = 0
    filtered = []

    while i < len(final_force) - window_size + 1:
        window_average = round(np.sum(final_force[i:i + window_size]) / window_size, 2)
        filtered.append(window_average)
        i += 1

    return filtered


import struct


def deserialize_io_states(raw_data):
    """
    Deserialize raw byte data into a structured IOStates object.

    Args:
    - raw_data (bytes): Raw byte data from IOStates message.

    Returns:
    - dict: Deserialized IOStates data as a dictionary.
    """

    # Initialize the index to read the raw data
    index = 0

    # 1. Deserialize Digital Inputs (18 pins, 1 byte per pin)
    digital_in_states = []
    for i in range(18):
        # Each pin is represented by a single byte (0 for low, 1 for high)
        state = struct.unpack_from('B', raw_data, index)[0]
        digital_in_states.append({"pin": i, "state": bool(state)})
        index += 1

    # 2. Deserialize Digital Outputs (18 pins, 1 byte per pin)
    digital_out_states = []
    for i in range(18):
        state = struct.unpack_from('B', raw_data, index)[0]
        digital_out_states.append({"pin": i, "state": bool(state)})
        index += 1

    # 3. Deserialize Analog Inputs (2 pins, 4 bytes per pin, as floats)
    analog_in_states = []
    for i in range(2):
        state = struct.unpack_from('f', raw_data, index)[0]  # 'f' is for 4-byte float
        analog_in_states.append({"pin": i, "domain": 1, "state": state})
        index += 4  # move forward by 4 bytes (size of float)

    # 4. Deserialize Analog Outputs (2 pins, 4 bytes per pin, as floats)
    analog_out_states = []
    for i in range(2):
        state = struct.unpack_from('f', raw_data, index)[0]  # 'f' is for 4-byte float
        analog_out_states.append({"pin": i, "domain": 0, "state": state})
        index += 4  # move forward by 4 bytes (size of float)

    # Create the structured output as a dictionary
    # return {
    #     "digital_in_states": digital_in_states,
    #     "digital_out_states": digital_out_states,
    #     "analog_in_states": analog_in_states,
    #     "analog_out_states": analog_out_states,
    #     "flag_states": []  # assuming flag_states is empty as in your sample data
    # }
    return analog_in_states


def unpack(buf):
    # Create a new message object (RobotState is the type)
    msg = IOStates()

    # Deserialize buffer into message object
    deserialize_message(buf, msg)

    return msg