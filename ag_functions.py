import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
from scipy.ndimage import median_filter, gaussian_filter
from scipy.signal import butter, filtfilt

import rosbag

# Median filter
def filter_force_m(variables, param):
    filtered = []
    for i in range(len(variables)):
        temp = median_filter(variables[i], param)
        filtered.append(temp)

    return filtered

# Calculate the elapsed time relative to the first timestamp in an array
def elapsed_time(variable, time_array):
    elapsed_t = [None] * len(variable)
    for i in range(len(variable)):
        elapsed_t[i] = (time_array[i] - time_array[0])  # Calculate difference from first time entry
    return elapsed_t

# Combine seconds and nanoseconds arrays into a single timestamp in seconds
def total_time(seconds, nseconds):
    time = []
    for i in range(len(seconds)):
        total = seconds[i] + (nseconds[i] / 1000000000)  # Convert nanoseconds to seconds and add to seconds
        time.append(total)
    return time

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
    # Initialize an empty list to store data from each message
    df = []

    # Create a reader instance to open the `.db3` file for reading
    with Reader('./' + name) as reader:
        # Iterate over each message in the bag file
        for connection, timestamp, rawdata in reader.messages():
            # Check if the message is from the wrench topic (force-torque sensor)
            if connection.topic == '/force_torque_sensor_broadcaster/wrench':
                # Deserialize the message data using CDR (Common Data Representation) format
                msg = deserialize_cdr(rawdata, connection.msgtype)

                # Extract force components along x, y, and z axes
                Fx = msg.wrench.force.x
                Fy = msg.wrench.force.y
                Fz = msg.wrench.force.z

                # Extract timestamp (seconds and nanoseconds) from message header
                nsecs = msg.header.stamp.nanosec
                secs = msg.header.stamp.sec

                # Compile extracted data into a list
                new_values = [Fx, Fy, Fz, nsecs, secs]
                # Append this list to the main data list
                df.append(new_values)

    # Save the accumulated data to a CSV file using the provided folder_name as the file name
    np.savetxt(folder_name + '.csv', df, delimiter=",")

    # Return the folder name as a confirmation of successful save
    return folder_name

# Reads a .db3 ros2 bag file and extracts relevant pressure data to a csv file
# NOT IN PICK_CLASSIFIER.PY
def db3_to_csv_p(folder_name):
    # Convert folder_name to a string for use as the file name
    name = str(folder_name)

    # Initialize lists to store pressure data from three different sensors
    pressure1 = []
    pressure2 = []
    pressure3 = []
    pressure = []

    # Define the topics corresponding to the pressure sensors
    topic1 = '/gripper/pressure/sc1'
    topic2 = '/gripper/pressure/sc2'
    topic3 = '/gripper/pressure/sc3'

    # Create a reader instance to open the `.db3` file for reading
    with Reader('./' + name) as reader:
        # Iterate over messages in the bag file
        for connection, timestamp, rawdata in reader.messages():
            # Check if the message is from the first pressure sensor topic
            if connection.topic == topic1:
                # Deserialize the message data using CDR format
                msg = deserialize_cdr(rawdata, connection.msgtype)
                # Extract the pressure data and append it to pressure1 list
                P1 = msg.data
                pressure1.append(P1)

            # Check if the message is from the second pressure sensor topic
            if connection.topic == topic2:
                # Deserialize the message data
                msg = deserialize_cdr(rawdata, connection.msgtype)
                # Extract the pressure data and append it to pressure2 list
                P2 = msg.data
                pressure2.append(P2)

            # Check if the message is from the third pressure sensor topic
            if connection.topic == topic3:
                # Deserialize the message data
                msg = deserialize_cdr(rawdata, connection.msgtype)
                # Extract the pressure data and append it to pressure3 list
                P3 = msg.data
                pressure3.append(P3)

    # Combine the three pressure sensor data into a single list of lists
    for i in range(len(pressure1)):
        new_pressure = [pressure1[i], pressure2[i], pressure3[i]]
        pressure.append(new_pressure)

    # Save the pressure data to a CSV file
    np.savetxt(folder_name + '_pressure.csv', pressure, delimiter=",")

    # Return the folder name as confirmation
    return folder_name

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