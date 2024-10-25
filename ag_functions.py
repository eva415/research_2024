import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
from scipy.ndimage import median_filter, gaussian_filter
from scipy.signal import butter, filtfilt

import rosbag


# function list
def filter_force_m(variables, param):
    # Median Filter
    filtered = []
    for i in range(len(variables)):
        temp = median_filter(variables[i], param)
        filtered.append(temp)

    return filtered


def filter_force_g(variables, param):
    # Median Filter
    filtered = []
    for i in range(len(variables)):
        temp = gaussian_filter(variables[i], sigma=param)
        filtered.append(temp)

    return filtered


def elapsed_time(variable, time_array):
    elapsedt = [None] * len(variable)
    for i in range(len(variable)):
        elapsedt[i] = (time_array[i] - time_array[0])
    return elapsedt


def total_time(seconds, nseconds):
    time = []
    for i in range(len(seconds)):
        total = seconds[i] + (nseconds[i] / 1000000000)
        time.append(total)
    return time


def bag_to_csv(i):
    file = str(i)
    bag = rosbag.Bag('./' + file + '.bag')
    topic = 'wrench'
    df = []

    for topic, msg, t in bag.read_messages(topics=topic):
        Fx = msg.wrench.force.x
        Fy = msg.wrench.force.y
        Fz = msg.wrench.force.z
        nsecs = msg.header.stamp.nsecs
        secs = msg.header.stamp.secs

        new_values = [Fx, Fy, Fz, nsecs, secs]
        df.append(new_values)

    np.savetxt(file + '.csv', df, delimiter=",")

    return file


def butter_lowpass_filter(data, cutoff, fs, order):
    nyq = 0.5 * fs  # Nyquist Frequency
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data, padlen=0)
    return y


def match_times(t1, t2, x1, x2):
    # extract all timestamps
    all_timestamps = sorted(set(t1).union(set(t2)))
    # initialize arrays
    new_x1 = np.zeros([len(all_timestamps), x1.shape[1]])
    new_x2 = np.zeros([len(all_timestamps), x2.shape[1]])
    # linear interpolation
    x1_num_cols = np.size(x1, 1)
    for i in range(x1_num_cols):
        new_x1[:, i] = np.interp(all_timestamps, t1, x1[:, i])

    x2_num_cols = np.size(x2, 1)
    for i in range(x2_num_cols):
        new_x2[:, i] = np.interp(all_timestamps, t2, x2[:, i])

    return new_x1, new_x2, all_timestamps


def flipping_data(cropped_x):
    new_x = []
    for i in range(len(cropped_x)):
        x = np.absolute(cropped_x[i] - cropped_x[0])
        new_x.append(x)

    return new_x


def bag_pressure(file):
    pressure1 = []
    pressure2 = []
    pressure3 = []
    pressure = []

    bag = rosbag.Bag('./' + file + '.bag')
    topic1 = '/gripper/pressure/sc1'
    topic2 = '/gripper/pressure/sc2'
    topic3 = '/gripper/pressure/sc3'

    for topic1, msg, t in bag.read_messages(topics=topic1):
        P1 = msg.data
        pressure1.append(P1)

    for topic2, msg, t in bag.read_messages(topics=topic2):
        P2 = msg.data
        pressure2.append(P2)

    for topic3, msg, t in bag.read_messages(topics=topic3):
        P3 = msg.data
        pressure3.append(P3)

    for i in range(len(pressure1)):
        new_pressure = [pressure1[i], pressure2[i], pressure3[i]]
        pressure.append(new_pressure)

    pressure_array = np.array(pressure)

    return pressure_array


def pressure_time(file):
    pt1s = []
    pt1n = []
    pt2s = []
    pt2n = []
    pt3s = []
    pt3n = []
    pts = []
    ptn = []

    bag = rosbag.Bag('./' + file + '.bag')
    topic1 = '/gripper/pressure/sc1'
    topic2 = '/gripper/pressure/sc2'
    topic3 = '/gripper/pressure/sc3'

    for topic1, msg, t in bag.read_messages(topics=topic1):
        t1s = t.secs
        t1n = t.nsecs
        pt1s.append(t1s)
        pt1n.append(t1n)

    for topic2, msg, t in bag.read_messages(topics=topic2):
        t2s = t.secs
        t2n = t.nsecs
        pt2s.append(t2s)
        pt2n.append(t2n)

    for topic3, msg, t in bag.read_messages(topics=topic3):
        t3s = t.secs
        t3n = t.nsecs
        pt3s.append(t3s)
        pt3n.append(t3n)

    for i in range(len(pt1s)):
        new = int(pt1s[i] + pt2s[i] + pt3s[i]) / 3
        pts.append(new)

    for i in range(len(pt1n)):
        new = int(pt1n[i] + pt2n[i] + pt3n[i]) / 3
        ptn.append(new)

    times = total_time(pts, ptn)
    etimes_pressure = elapsed_time(times, times)
    return etimes_pressure


def db3_to_csv_f(folder_name):
    name = str(folder_name)
    df = []
    # create reader instance and open for reading
    with Reader('./' + name) as reader:
        # iterate over messages
        for connection, timestamp, rawdata in reader.messages():
            if connection.topic == '/force_torque_sensor_broadcaster/wrench':
                msg = deserialize_cdr(rawdata, connection.msgtype)
                Fx = msg.wrench.force.x
                Fy = msg.wrench.force.y
                Fz = msg.wrench.force.z
                nsecs = msg.header.stamp.nanosec
                secs = msg.header.stamp.sec

                new_values = [Fx, Fy, Fz, nsecs, secs]
                df.append(new_values)

    np.savetxt(folder_name + '.csv', df, delimiter=",")

    return folder_name


def db3_to_csv_p(folder_name):
    name = str(folder_name)
    pressure1 = []
    pressure2 = []
    pressure3 = []
    pressure = []
    topic1 = '/gripper/pressure/sc1'
    topic2 = '/gripper/pressure/sc2'
    topic3 = '/gripper/pressure/sc3'
    # create reader instance and open for reading
    with Reader('./' + name) as reader:
        # iterate over messages
        for connection, timestamp, rawdata in reader.messages():
            if connection.topic == topic1:
                msg = deserialize_cdr(rawdata, connection.msgtype)
                P1 = msg.data
                pressure1.append(P1)

            if connection.topic == topic2:
                msg = deserialize_cdr(rawdata, connection.msgtype)
                P2 = msg.data
                pressure2.append(P2)

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