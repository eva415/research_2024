import os
import csv
from rclpy.serialization import deserialize_message
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
from scipy.signal import butter, filtfilt
from rosidl_runtime_py.utilities import get_message
import matplotlib.pyplot as plt
import numpy as np
import scipy as scipy
from scipy.ndimage import median_filter

INDEX_OG = 0  # set to 100 generally, 0 for exception cases

failed_bag_files = ["final_approach_and_pick_20251029_172728",
                    "final_approach_and_pick_20251029_172946",
                    "final_approach_and_pick_20251029_173938",
                    "final_approach_and_pick_20251029_174404",
                    "final_approach_and_pick_20251029_180739",
                    "final_approach_and_pick_20251030_114539",
                    "final_approach_and_pick_20251030_114927",
                    "final_approach_and_pick_20251030_130515",
                    "final_approach_and_pick_20251030_132429",
                    "final_approach_and_pick_20251029_165052",
                    "final_approach_and_pick_20251030_150419"]

failed_corresponding_slip_times = [18.51,
                                    12.74,
                                    15.33,
                                    4.08,
                                    3.82,
                                    12.20,
                                    14.80,
                                    9.64,
                                    12.43,
                                    5.4,
                                    3.67]

# successful_bag_files = ["final_approach_and_pick_20251028_143901",
# "final_approach_and_pick_20251028_171000",
# "final_approach_and_pick_20251028_171431",
# "final_approach_and_pick_20251029_164443",
# "final_approach_and_pick_20251029_164729",
# "final_approach_and_pick_20251029_165334",
# "final_approach_and_pick_20251029_170049",
# "final_approach_and_pick_20251029_170153",
# "final_approach_and_pick_20251029_170659",
# "final_approach_and_pick_20251029_170813",
# "final_approach_and_pick_20251029_171431",
# "final_approach_and_pick_20251029_171756",
# "final_approach_and_pick_20251029_171959",
# "final_approach_and_pick_20251029_172251",
# "final_approach_and_pick_20251029_172622",
# "final_approach_and_pick_20251029_173433",
# "final_approach_and_pick_20251029_173557",
# "final_approach_and_pick_20251029_173742",
# "final_approach_and_pick_20251029_174813",
# "final_approach_and_pick_20251029_174943",
# "final_approach_and_pick_20251029_175156",
# "final_approach_and_pick_20251029_175542",
# "final_approach_and_pick_20251029_175730",
# "final_approach_and_pick_20251029_180355",
# "final_approach_and_pick_20251029_180549",
# "final_approach_and_pick_20251030_115312",
# "final_approach_and_pick_20251030_130354",
# "final_approach_and_pick_20251030_130656",
# "final_approach_and_pick_20251030_130808",
# "final_approach_and_pick_20251030_130932",
# "final_approach_and_pick_20251030_131112",
# "final_approach_and_pick_20251030_131440",
# "final_approach_and_pick_20251030_131700",
# "final_approach_and_pick_20251030_131808",
# "final_approach_and_pick_20251030_132329",
# "final_approach_and_pick_20251030_132747",
# "final_approach_and_pick_20251030_132850",
# "final_approach_and_pick_20251030_133135",
# "final_approach_and_pick_20251030_133243",
# "final_approach_and_pick_20251030_133504",
# "final_approach_and_pick_20251030_133640",
# "final_approach_and_pick_20251030_133814",
# "final_approach_and_pick_20251030_134038",
# "final_approach_and_pick_20251030_134512",
# "final_approach_and_pick_20251030_135117",
# "final_approach_and_pick_20251030_135343",
# "final_approach_and_pick_20251030_135620",
# "final_approach_and_pick_20251030_135801",
# "final_approach_and_pick_20251030_135929",
# "final_approach_and_pick_20251030_140049",
# "final_approach_and_pick_20251030_140219",
# "final_approach_and_pick_20251030_140350",
# "final_approach_and_pick_20251030_140546",
# "final_approach_and_pick_20251030_140742",
# "final_approach_and_pick_20251030_140900",
# "final_approach_and_pick_20251030_141836",
# "final_approach_and_pick_20251030_142154",
# "final_approach_and_pick_20251030_142303",
# "final_approach_and_pick_20251030_142419",
# "final_approach_and_pick_20251030_142918",
# "final_approach_and_pick_20251030_143031",
# "final_approach_and_pick_20251030_143452",
# "final_approach_and_pick_20251030_143657",
# "final_approach_and_pick_20251030_143827",
# "final_approach_and_pick_20251030_144444",
# "final_approach_and_pick_20251030_144548",
# "final_approach_and_pick_20251030_144845",
# "final_approach_and_pick_20251030_145001",
# "final_approach_and_pick_20251030_145507",
# "final_approach_and_pick_20251030_145700",
# "final_approach_and_pick_20251030_150019",
# "final_approach_and_pick_20251030_150803"]
# # print(f"files len: {len(successful_bag_files)}")

# successful_corresponding_pick_times = [16.12,
# 10.34,
# 12.12,
# 11.46,
# 5.64,
# 5.15,
# 7.76,
# 9.96,
# 7.26,
# 12.49,
# 5.88,
# 14.47,
# 14.1,
# 19.04,
# 18.08,
# 9.05,
# 16.4,
# 4.91,
# 10.76,
# 5.64,
# 11.36,
# 4.64,
# 13.61,
# 4.72,
# 8.5,
# 4.12,
# 8.35,
# 2.64,
# 7.21,
# 8.23,
# 7.42,
# 4.21,
# 8.35,
# 10.54,
# 9.93,
# 7.11,
# 8.7,
# 4.19,
# 10.61,
# 11.55,
# 5.73,
# 4.09,
# 14.85,
# 8.05,
# 4.46,
# 4.42,
# 10.7,
# 3.29,
# 3.95,
# 3.68,
# 6.36,
# 8.11,
# 8.5,
# 7.12,
# 10.14,
# 5.51,
# 4.35,
# 4.99,
# 3.35,
# 11.56,
# 9.29,
# 6.52,
# 4.8,
# 1.91,
# 7.51,
# 8.38,
# 5.85,
# 5.65,
# 10.47,
# 9.9,
# 4.26,
# 8.44]
# # print(f"times len: {len(successful_corresponding_pick_times)}")

# build mapping for quick lookup
failed_map = dict(zip(failed_bag_files, failed_corresponding_slip_times))
# successful_map = dict(zip(successful_bag_files, successful_corresponding_pick_times))

# helper: detect failed bag and return slip time or None
def get_failed_slip_time(bag_path):
    base = os.path.basename(str(bag_path))
    for k, t in failed_map.items():
        if k in base:
            return t
    return None

# def get_successful_pick_time(bag_path):
#     base = os.path.basename(str(bag_path))
#     for k, t in successful_map.items():
#         if k in base:
#             return t
#     return None

# ---------- helpers ----------
def elapsed_time(time_array):
    time_array = np.array(time_array)
    if time_array.size == 0:
        print("Error: time_array is empty.")
        return None
    return time_array - time_array[0]

def total_time(seconds, nseconds):
    seconds = np.array(seconds, dtype=np.float64)
    nseconds = np.array(nseconds, dtype=np.float64)
    return seconds + (nseconds / 1e9)

def butter_lowpass_filter(data, cutoff=50, fs=500., order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data, padlen=0)
    return y

def filter_force(variables, param):
    filtered = []
    for var in variables:
        filtered.append(median_filter(var, param))
    return filtered

def _write_csv_rows(path, rows, header=None):
    """Utility to write rows (list of lists) to CSV path. Overwrites existing file."""
    with open(path, 'w', newline='') as fh:
        writer = csv.writer(fh)
        if header:
            writer.writerow(header)
        writer.writerows(rows)

def _load_numeric_csv_allow_classification(path, expected_data_cols):
    """
    Loads a numeric CSV which may optionally have an extra final column 'classification'.
    Expects last two columns to be secs, nsecs. Returns:
      data_arr (N x expected_data_cols)  # numeric sensor data (without classification)
      secs_arr (N,)
      nsecs_arr (N,)
      classification_arr (N,) or None   # numeric classification if present (0/2)
    """
    try:
        arr = np.loadtxt(path, delimiter=',', dtype=float)
    except Exception as e:
        # Could be headered or empty; try a robust csv.reader fallback
        rows = []
        with open(path, 'r') as fh:
            reader = csv.reader(fh)
            for r in reader:
                if not r:
                    continue
                # try to convert all to float, skip if fail
                try:
                    rows.append([float(x) for x in r])
                except Exception:
                    continue
        if not rows:
            return None, None, None, None
        arr = np.array(rows, dtype=float)

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    # last two columns should be secs, nsecs
    if arr.shape[1] < expected_data_cols + 2:
        # unexpected
        return None, None, None, None

    # If there is an extra column beyond expected_data_cols + 2, it's the classification
    if arr.shape[1] == expected_data_cols + 2:
        data = arr[:, :expected_data_cols]
        secs = arr[:, -2]
        nsecs = arr[:, -1]
        classification = None
    else:
        # arr.shape[1] >= expected_data_cols + 3
        # We assume the classification is the last column
        data = arr[:, :expected_data_cols]
        secs = arr[:, -2]
        nsecs = arr[:, -1]
        classification = arr[:, -3] if arr.shape[1] == expected_data_cols + 3 else arr[:, -3]  # last-but-two
        # Note: for safety we choose last-but-two if there are extra columns; but in our writer we only add one col

    return data, secs, nsecs, classification

# ---------- db3 -> csv (modified to append numeric classification per-row when bag is in failed list) ----------

def db3_to_csv_f(folder_name):
    name = str(folder_name)
    df = []

    with Reader(name) as reader:
        for connection, timestamp, rawdata in reader.messages():
            if connection.topic == '/force_torque_sensor_broadcaster/wrench':
                msg = deserialize_cdr(rawdata, connection.msgtype)
                Fx = float(msg.wrench.force.x)
                Fy = float(msg.wrench.force.y)
                Fz = float(msg.wrench.force.z)
                secs = float(msg.header.stamp.sec)
                nsecs = float(msg.header.stamp.nanosec)
                df.append([Fx, Fy, Fz, secs, nsecs])

    out_path = name + 'force.csv'
    if not df:
        print("No force data found in the specified topic.")
        return None

    slip_time = get_failed_slip_time(name)
    # slip_time = get_successful_pick_time(name)

    if slip_time is not None:
        # compute elapsed times
        secs_arr = np.array([r[3] for r in df], dtype=float)
        nsecs_arr = np.array([r[4] for r in df], dtype=float)
        total = secs_arr + nsecs_arr / 1e9
        elapsed = total - total[0]
        rows = []
        # append classification as last column: 0 for P (before slip_time), 2 for F (at/after)
        for i, r in enumerate(df):
            cls = 3 if elapsed[i] < slip_time else 2 # CHANGE HERE TO MAKE 0 1 2 OR 3
            # sensor cols (Fx,Fy,Fz), secs, nsecs, classification appended as last column
            rows.append([r[0], r[1], r[2], r[3], r[4], cls])
        _write_csv_rows(out_path, rows)
    else:
        # no classification: Fx, Fy, Fz, secs, nsecs
        _write_csv_rows(out_path, df)

    return name + 'force'


def db3_to_csv_p(folder_name):
    name = str(folder_name)
    df = []

    with Reader(name) as reader:
        for connection, timestamp, rawdata in reader.messages():
            if connection.topic == '/vacuum_pressure':
                msg_type = get_message('std_msgs/msg/Float32')
                msg = deserialize_message(rawdata, msg_type)
                P = float(msg.data)
                secs = float(timestamp // 1_000_000_000)
                nsecs = float(timestamp % 1_000_000_000)
                df.append([P, secs, nsecs])

    out_path = name + 'pressure.csv'
    if not df:
        print("No pressure data found in the specified topic.")
        return None

    slip_time = get_failed_slip_time(name)
    # slip_time = get_successful_pick_time(name)

    if slip_time is not None:
        secs_arr = np.array([r[1] for r in df], dtype=float)
        nsecs_arr = np.array([r[2] for r in df], dtype=float)
        total = secs_arr + nsecs_arr / 1e9
        elapsed = total - total[0]
        rows = []
        for i, r in enumerate(df):
            cls = 3 if elapsed[i] < slip_time else 2
            # P, secs, nsecs, class appended last
            rows.append([r[0], r[1], r[2], cls])
        _write_csv_rows(out_path, rows)
    else:
        _write_csv_rows(out_path, df)

    return name + 'pressure'


def db3_to_csv_tof(folder_name):
    name = str(folder_name)
    df = []

    with Reader(name) as reader:
        for connection, timestamp, rawdata in reader.messages():
            if connection.topic == '/tof_sensor_data':
                msg_type = get_message('std_msgs/msg/Int32')
                msg = deserialize_message(rawdata, msg_type)
                tof = int(msg.data)
                secs = float(timestamp // 1_000_000_000)
                nsecs = float(timestamp % 1_000_000_000)
                df.append([float(tof), secs, nsecs])

    out_path = name + 'tof.csv'
    if not df:
        print("No TOF data found in this bag.")
        return None

    slip_time = get_failed_slip_time(name)
    # slip_time = get_successful_pick_time(name)
    if slip_time is not None:
        secs_arr = np.array([r[1] for r in df], dtype=float)
        nsecs_arr = np.array([r[2] for r in df], dtype=float)
        total = secs_arr + nsecs_arr / 1e9
        elapsed = total - total[0]
        rows = []
        for i, r in enumerate(df):
            cls = 3 if elapsed[i] < slip_time else 2
            rows.append([r[0], r[1], r[2], cls])
        _write_csv_rows(out_path, rows)
    else:
        _write_csv_rows(out_path, df)

    return name + 'tof'


def db3_to_csv_flex(folder_name):
    name = str(folder_name)
    rows = []

    with Reader(name) as reader:
        for connection, timestamp, rawdata in reader.messages():
            if connection.topic == '/flex_sensor_data':
                msg_type = get_message('std_msgs/msg/Float32MultiArray')
                msg = deserialize_message(rawdata, msg_type)
                flex_values = np.array(msg.data, dtype=float)
                # collapses to single reading (L2 norm)
                flex_norm = float(np.linalg.norm(flex_values))
                secs  = float(timestamp // 1_000_000_000)
                nsecs = float(timestamp % 1_000_000_000)
                # store secs, nsecs, all flex channels, flex_norm
                rows.append([secs, nsecs] + flex_values.tolist() + [flex_norm])

    if not rows:
        print("No flex sensor data found in this bag.")
        return None

    # pad to equal length
    max_len = max(len(r) for r in rows)
    padded = [r + [np.nan] * (max_len - len(r)) for r in rows]

    # determine slip_time and build final rows with classification as last column
    slip_time = get_failed_slip_time(name)
    # slip_time = get_successful_pick_time(name)

    # compute secs/nsecs arrays from padded rows
    secs_arr = np.array([r[0] for r in padded], dtype=float)
    nsecs_arr = np.array([r[1] for r in padded], dtype=float)
    total = secs_arr + nsecs_arr / 1e9
    elapsed = total - total[0]

    final_rows = []
    for i, r in enumerate(padded):
        cls = 0
        if slip_time is not None:
            cls = 3 if elapsed[i] < slip_time else 2
        # append classification as last column
        final_rows.append(r + [cls])

    # Build header (sec, nsec, flex_0,..., flex_N, flex_norm, class)
    num_cols = len(final_rows[0])
    # original flex_data columns = max_len; first two are sec,nsec; last is flex_norm
    num_flex_channels = max_len - 3  # subtract sec,nsec,flex_norm
    header = ["sec", "nsec"] + [f"flex_{i}" for i in range(num_flex_channels)] + ["flex_norm", "class"]

    # write CSV with header
    _write_csv_rows(name + "_flex_norm.csv", final_rows, header=header)

    return name + "_flex_norm"

# ---------- loaders: these return arrays compatible with existing code (they drop classification column) ----------

def return_tof_array(filename):
    file_t = db3_to_csv_tof(filename)
    csv_path = './' + file_t + '.csv'
    # expected_data_cols before secs,nsecs = 1
    data_t, raw_sec_array_t, raw_nsec_array_t, classification = _load_numeric_csv_allow_classification(csv_path, expected_data_cols=1)
    if data_t is None:
        return None, None

    total_tof_time = total_time(raw_sec_array_t, raw_nsec_array_t)
    elapsed_tof_time = elapsed_time(total_tof_time)

    raw_tof_values = data_t[:, 0].squeeze()   # (remove timestamps)

    return raw_tof_values, elapsed_tof_time, classification

def return_force_array(filename):
    file_f = db3_to_csv_f(filename)
    data_f = np.loadtxt('./' + file_f + '.csv', dtype="float", delimiter=',')
    raw_sec_array_f = data_f[:, -3]
    raw_nsec_array_f = data_f[:, -2]
    total_time_force = total_time(raw_sec_array_f, raw_nsec_array_f)
    elapsed_time_force = elapsed_time(total_time_force)
    raw_force_array = data_f[:, :-3]
    fz_force_array = -raw_force_array[:, 2]
    classification = data_f[:, -1]
    return raw_force_array, fz_force_array, elapsed_time_force, classification

def return_pressure_array(filename):
    file_p = db3_to_csv_p(filename)
    csv_path = './' + file_p + '.csv'
    data_p, raw_sec_array_p, raw_nsec_array_p, classification = _load_numeric_csv_allow_classification(csv_path, expected_data_cols=1)
    if data_p is None:
        return None, None

    total_time_pressure = total_time(raw_sec_array_p, raw_nsec_array_p)
    elapsed_time_pressure = elapsed_time(total_time_pressure)
    raw_pressure_array = data_p[:, 0]
    return raw_pressure_array, elapsed_time_pressure, classification

def return_flex_array(filename, smoothing_method='median', param=5):
    """
    Returns smoothed flex norm and elapsed time.
    smoothing_method: 'median' or 'butter'
    param: window size for median, cutoff freq for butter
    """
    # Convert rosbag → CSV (your flex_norm version)
    file_flex = db3_to_csv_flex(filename)

    if file_flex is None:
        print("No flex file returned.")
        return None, None

    csv_path = './' + file_flex + '.csv'
    # load full numeric table (header present in flex file)
    try:
        data_flex = np.loadtxt(csv_path, dtype=float, delimiter=',', skiprows=1)
    except Exception:
        # fallback robust reader
        rows = []
        with open(csv_path, 'r') as fh:
            reader = csv.reader(fh)
            # skip header
            next(reader, None)
            for r in reader:
                if not r:
                    continue
                try:
                    rows.append([float(x) for x in r])
                except Exception:
                    continue
        data_flex = np.array(rows, dtype=float)

    if data_flex.ndim == 1:
        data_flex = data_flex.reshape(1, -1)

    # Last numeric column before 'class' is flex_norm (class is last)
    flex_norm = data_flex[:, -2]  # second-last column
    # Extract timestamps
    raw_sec_array_flex  = data_flex[:, 0]
    raw_nsec_array_flex = data_flex[:, 1]

    # Compute time arrays
    total_flex_time   = total_time(raw_sec_array_flex, raw_nsec_array_flex)
    elapsed_flex_time = elapsed_time(total_flex_time)

    # --- Smooth flex signal ---
    if smoothing_method == 'median':
        flex_norm_smooth = median_filter(flex_norm, size=param)
    elif smoothing_method == 'butter':
        flex_norm_smooth = butter_lowpass_filter(flex_norm, cutoff=param, fs=500, order=2)
    else:
        flex_norm_smooth = flex_norm  # no smoothing
    classification = data_flex[:, -1]
    return flex_norm_smooth, elapsed_flex_time, classification

# ---------- rest of your code unchanged (classifier, plotting, etc.) ----------
# (I reproduce your functions below unchanged so the file is complete.)

# GENERIC PICK CLASSIFIER
def picking_type_classifier_force_multi(
        force,
        sensors,                # dict: {"pressure": array, "tof": array, "flex": array}
        thresholds,             # dict: {"pressure": X, "tof": Y, "flex": Z}
        combination_logic,      # "OR", "AND", or custom lambda
        force_threshold,
        force_change_threshold):

    def moving_average(final_force):
        window_size = 5
        return [
            round(np.sum(final_force[k:k+window_size]) / window_size, 2)
            for k in range(len(final_force) - window_size + 1)
        ]

    wait_sync = 10
    count = 0
    pick_type = "Unclassified"
    pick_start = None
    i = 10

    # Ensure sensors have matching length
    min_len = min([len(force)] + [len(v) for v in sensors.values()])
    
    while i < min_len - 10:

        cropped_force = force[i - 10:i]
        filtered_force = moving_average(cropped_force)

        # Compute derivative --------------------------
        backwards_diff = []
        h = 2
        for j in range(2*h, len(filtered_force)):
            diff = ((3 * filtered_force[j])
                    - (4 * filtered_force[j - h])
                    + filtered_force[j - 2*h]) / (2 * h)
            backwards_diff.append(diff)
        dF = np.average(backwards_diff) if backwards_diff else 0.0

        # Compute sensor averages ---------------------
        avg_values = {}
        for key, arr in sensors.items():
            avg_values[key] = np.average(arr[i - 10:i])

        # Decide failure based on OR / AND or custom logic
        def failure_condition():
            if combination_logic == "OR":
                return any(avg_values[s] >= thresholds[s] for s in avg_values)
            elif combination_logic == "AND":
                return all(avg_values[s] >= thresholds[s] for s in avg_values)
            else:
                # custom boolean lambda(avg_values, thresholds)
                return combination_logic(avg_values, thresholds)

        # Detect event --------------------------------
        if filtered_force and filtered_force[0] <= force_threshold:

            # → Successful pick path
            if dF >= force_change_threshold and not failure_condition():

                if pick_start is None:
                    pick_start = i

                if count < wait_sync:
                    count += 1
                    pick_type = "Successful"
                    i += 1
                    continue
                else:
                    # print(f"Apple picked! dF/dt:{dF}, sensors:{avg_values}, idx:{pick_start}")
                    return "Successful", pick_start

            # → Failed pick path
            elif failure_condition():
                # print(f"FAILED pick. dF/dt:{dF}, sensors:{avg_values}, idx:{i}")
                return "Failed", i

        i += 1

    return pick_type, pick_start if pick_start is not None else i

def shade_failed(ax, time, class_array):
    """Shade red where class == 1."""
    fail_regions = class_array == 1
    # Find contiguous regions
    idx = np.where(fail_regions)[0]

    if len(idx) == 0:
        return  # nothing to shade

    # Split into continuous segments
    splits = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)

    for segment in splits:
        start = time[segment[0]]
        end   = time[segment[-1]]
        ax.axvspan(start, end, color='red', alpha=0.2)


def picking_type_classifier_f_any(force, pressure, tof, flex,
                                  pressure_th, tof_th, flex_th,
                                  force_th, force_change_th):
    sensors = {"pressure": pressure, "tof": tof, "flex": -flex}
    thresholds = {"pressure": pressure_th, "tof": tof_th, "flex": -flex_th}
    return picking_type_classifier_force_multi(
        force, sensors, thresholds, combination_logic="OR",
        force_threshold=force_th,
        force_change_threshold=force_change_th
    )

def process_file_and_graph_pick_analysis(filename, pressure_threshold, force_threshold, force_change_threshold, tof_threshold, flex_threshold):
    raw_f_arr, f_arr, etime_force, force_class = return_force_array(filename)
    p_arr, etimes_pressure, _ = return_pressure_array(filename)
    tof_arr, etimes_tof, _ = return_tof_array(filename)
    flex_arr, etimes_flex, _ = return_flex_array(filename)

    # resample
    tof_arr_new = scipy.signal.resample(tof_arr, len(f_arr))
    flex_arr_new = scipy.signal.resample(flex_arr, len(f_arr))
    p_arr_new = scipy.signal.resample(p_arr, len(f_arr))
    filtered_force = filter_force([f_arr], 21)[0]
    force_class = force_class.astype(int)
    
    # Find start index where pressure first <= -56
    start_index_candidates = np.where(p_arr_new <= -56)[0]
    if len(start_index_candidates) == 0:
        print("Pressure never reaches -56, starting from the beginning.")
        start_index = 0
    else:
        start_index = start_index_candidates[0]+INDEX_OG

    # Central difference for derivative
    backwards_diff = []
    h = 2
    for j in range(2*h, len(filtered_force)):
        diff = ((3*filtered_force[j]) - (4*filtered_force[j-h]) + filtered_force[j-2*h]) / (2*h)
        backwards_diff.append(diff)
    low_bdiff = butter_lowpass_filter(backwards_diff)

    # Pass the start_index to the classifier
    # calling force + (pressure or tof or flex)
    pick_type, pick_i = picking_type_classifier_f_any(
        filtered_force[start_index:-10],      # force
        p_arr_new[start_index: -10],            # pressure
        tof_arr_new[start_index:-10],           # tof
        flex_arr_new[start_index:-10],         # flex
        pressure_threshold,             # pressure_th
        tof_threshold,                # tof_th
        flex_threshold,                     # flex_th
        force_threshold,                   # force_th
        force_change_threshold             # force_change_th
    )

    # TODO: call random forest classifier here


    # Adjust pick_i to match the full array indexing
    pick_i += start_index

    # Convert pick index to time
    pick_time = etime_force[pick_i]

    # Plot results
    fig, ax = plt.subplots(4, 1, figsize=(12, 12))
    
    ax[0].plot(etime_force[:len(low_bdiff)], low_bdiff, label='Filtered dF/dt', color='black')
    ax[0].axhline(force_change_threshold, color='blue', linestyle='--', label='Force Change Threshold')
    ax[0].axvline(pick_time, color='red', linestyle='--', label=f'Pick Event ({pick_time:.2f}s)')
    ax[0].axvspan(etime_force[start_index], etime_force[-1], color='yellow', alpha=0.2, label='Analysis Region')
    ax[0].set_title("Force Derivative")
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("dF/dt")
    ax[0].legend(loc='upper left')
    
    ax[1].plot(etime_force, p_arr_new, label='Pressure', color='red')
    ax[1].axhline(pressure_threshold, color='blue', linestyle='--', label='Pressure Threshold')
    ax[1].axvline(pick_time, color='red', linestyle='--', label=f'Pick Event ({pick_time:.2f}s)')
    ax[1].axvspan(etime_force[start_index], etime_force[-1], color='yellow', alpha=0.2, label='Analysis Region')
    ax[1].set_title("Gripper Pressure")
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Pressure")
    ax[1].legend(loc='upper left')

    ax[3].plot(etime_force, tof_arr_new, label='ToF', color='red')
    ax[3].axhline(tof_threshold, color='blue', linestyle='--', label='Pressure Threshold')
    ax[3].axvline(pick_time, color='red', linestyle='--', label=f'Pick Event ({pick_time:.2f}s)')
    ax[3].axvspan(etime_force[start_index], etime_force[-1], color='yellow', alpha=0.2, label='Analysis Region')
    ax[3].set_title("Gripper Distance to Apple (ToF)")
    ax[3].set_xlabel("Time (s)")
    ax[3].set_ylabel("Time of Flight Measurement (mm)")
    ax[3].legend(loc='upper left')

    ax[2].plot(etime_force, flex_arr_new, label='Flex', color='red')
    ax[2].axhline(flex_threshold, color='blue', linestyle='--', label='Normed Flex Threshold')
    ax[2].axvline(pick_time, color='red', linestyle='--', label=f'Pick Event ({pick_time:.2f}s)')
    ax[2].axvspan(etime_force[start_index], etime_force[-1], color='yellow', alpha=0.2, label='Analysis Region')
    ax[2].set_title("Normalized Flex Sensor Reading (degrees)")
    ax[2].set_xlabel("Time (s)")
    ax[2].set_ylabel("Normed Flex Measurement (degrees)")
    ax[2].legend(loc='upper left')

    shade_failed(ax[0], etime_force[start_index:], force_class[start_index:])
    shade_failed(ax[1], etime_force[start_index:], force_class[start_index:])
    shade_failed(ax[2], etime_force[start_index:], force_class[start_index:])
    shade_failed(ax[3], etime_force[start_index:], force_class[start_index:])
    # print(f"etime_force len: {len(etime_force)}, force_class len: {len(force_class)}")
    print(f"force_class unique values: {np.unique(force_class)}")
        
    plt.suptitle(f"Bag: {filename}\nPick Classification: {pick_type}\n(Actual Classification: Failed) ", fontsize=16)
    # plt.suptitle(f"Bag: {filename}\nPick Classification: {pick_type}\n(Actual Classification: Successful) ", fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.95, 0.95])
    # plt.show()

    # print(f"\nBag: {filename}\nPick Classification: {pick_type}\n(Actual Classification: Successful) ")
    print(f"\nBag: {filename}\nPick Classification: {pick_type}\n(Actual Classification: Failed) ")