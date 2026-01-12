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

def db3_to_csv_f(folder_name):
    name = str(folder_name)
    df = []

    with Reader(name) as reader:
        for connection, timestamp, rawdata in reader.messages():
            if connection.topic == '/force_torque_sensor_broadcaster/wrench':
                msg = deserialize_cdr(rawdata, connection.msgtype)
                Fx = msg.wrench.force.x
                Fy = msg.wrench.force.y
                Fz = msg.wrench.force.z
                secs = msg.header.stamp.sec
                nsecs = msg.header.stamp.nanosec
                df.append([Fx, Fy, Fz, secs, nsecs])

    if df:
        np.savetxt(name + 'force.csv', np.array(df), delimiter=",")
    else:
        print("No force data found in the specified topic.")
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
                secs = timestamp // 1_000_000_000
                nsecs = timestamp % 1_000_000_000
                df.append([P, secs, nsecs])

    if df:
        np.savetxt(name + 'pressure.csv', np.array(df), delimiter=",")
    else:
        print("No pressure data found in the specified topic.")
    return name + 'pressure'

def db3_to_csv_tof(folder_name):
    name = str(folder_name)
    df = []

    with Reader(name) as reader:
        for connection, timestamp, rawdata in reader.messages():
            if connection.topic == '/tof_sensor_data':    # <--- TOF topic from your metadata
                msg_type = get_message('std_msgs/msg/Int32')
                msg = deserialize_message(rawdata, msg_type)

                tof = int(msg.data)

                # Timestamp reconstruction from rosbag2 (just like pressure)
                secs = timestamp // 1_000_000_000
                nsecs = timestamp % 1_000_000_000

                df.append([tof, secs, nsecs])

    if df:
        np.savetxt(name + 'tof.csv', np.array(df), delimiter=",")
    else:
        print("⚠ No TOF data found in this bag.")

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

                # Your method → single collapse indicator
                flex_norm = np.linalg.norm(flex_values)

                secs  = timestamp // 1_000_000_000
                nsecs = timestamp % 1_000_000_000

                # Store timestamp, all raw flex channels, and the L2 norm
                rows.append([secs, nsecs] + flex_values.tolist() + [flex_norm])

    if not rows:
        print("No flex sensor data found in this bag.")
        return None

    # Convert to array and pad uneven rows (unlikely but safe)
    max_len = max(len(r) for r in rows)
    padded = [r + [np.nan] * (max_len - len(r)) for r in rows]
    data = np.array(padded)

    # Build header
    num_flex = max_len - 3  # subtract time + flex_norm
    header = ["sec", "nsec"] + [f"flex_{i}" for i in range(num_flex - 1)] + ["flex_norm"]

    out_path = name + "_flex_norm.csv"

    np.savetxt(out_path, data, delimiter=",", header=",".join(header), comments="")
    # print(f"Saved flex norm CSV: {out_path}")

    return out_path

def return_tof_array(filename):
    file_t = db3_to_csv_tof(filename)
    data_t = np.loadtxt('./' + file_t + '.csv', dtype=float, delimiter=',')

    raw_sec_array_t = data_t[:, -2]
    raw_nsec_array_t = data_t[:, -1]

    total_tof_time = total_time(raw_sec_array_t, raw_nsec_array_t)
    elapsed_tof_time = elapsed_time(total_tof_time)

    raw_tof_values = data_t[:, :-2].squeeze()   # (remove timestamps)

    return raw_tof_values, elapsed_tof_time

def return_force_array(filename):
    file_f = db3_to_csv_f(filename)
    data_f = np.loadtxt('./' + file_f + '.csv', dtype="float", delimiter=',')
    raw_sec_array_f = data_f[:, -2]
    raw_nsec_array_f = data_f[:, -1]
    total_time_force = total_time(raw_sec_array_f, raw_nsec_array_f)
    elapsed_time_force = elapsed_time(total_time_force)
    raw_force_array = data_f[:, :-2]
    fz_force_array = -raw_force_array[:, 2]
    return raw_force_array, fz_force_array, elapsed_time_force

def return_pressure_array(filename):
    file_p = db3_to_csv_p(filename)
    data_p = np.loadtxt('./' + file_p + '.csv', dtype="float", delimiter=',')
    raw_sec_array_p = data_p[:, -2]
    raw_nsec_array_p = data_p[:, -1]
    total_time_pressure = total_time(raw_sec_array_p, raw_nsec_array_p)
    elapsed_time_pressure = elapsed_time(total_time_pressure)
    raw_pressure_array = data_p[:, :-2]
    return raw_pressure_array, elapsed_time_pressure

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

    # Load CSV
    data_flex = np.loadtxt('./' + file_flex, dtype=float, delimiter=',', skiprows=1)

    # Last column = flex_norm
    flex_norm = data_flex[:, -1]

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

    return flex_norm_smooth, elapsed_flex_time

def picking_type_classifier_f_p(force, pressure, pressure_threshold, force_threshold, force_change_threshold):
    def moving_average(final_force):
        window_size = 5
        filtered = []
        for k in range(len(final_force) - window_size + 1):
            filtered.append(round(np.sum(final_force[k:k + window_size]) / window_size, 2))
        return filtered

    count = 0
    wait_for_pressure_sync = 10
    pick_type = 'Unclassified'
    i = 10
    pick_start = None   # <-- record index where spike was first detected

    while i < len(force) - 10:
        cropped_force = force[i - 10:i]
        filtered_force = moving_average(cropped_force)
        cropped_pressure = pressure[i - 10:i]
        avg_pressure = np.average(cropped_pressure)

        # compute backward derivative on filtered_force (keep same logic)
        backwards_diff = []
        h = 2
        for j in range(2 * h, len(filtered_force)):
            diff = ((3 * filtered_force[j]) - (4 * filtered_force[j - h]) + filtered_force[j - 2*h]) / (2 * h)
            backwards_diff.append(diff)
        cropped_backward_diff = np.average(backwards_diff) if backwards_diff else 0.0

        if filtered_force and filtered_force[0] <= force_threshold:
            if cropped_backward_diff >= force_change_threshold and avg_pressure < pressure_threshold:
                # first time we see the spike: record the index
                if pick_start is None:
                    pick_start = i   # record detection index (sample index)
                    # optionally, if you prefer timestamp corresponding to the center of the cropped window:
                    # pick_start = i - 5

                # waiting/debounce window: increment counter and advance time
                if count < wait_for_pressure_sync:
                    count += 1
                    pick_type = 'Successful'  # optimistic until pressure spikes
                    # print("waiting... (count = {})".format(count))
                    i += 1   # IMPORTANT: advance time so pressure samples can change
                    continue
                else:
                    # waiting finished with no pressure spike -> success at pick_start
                    pick_type = 'Successful'
                    print(f'Apple picked! dF/dt: {cropped_backward_diff}, Pressure: {avg_pressure}, detection_index: {pick_start}')
                    return pick_type, pick_start
            elif avg_pressure >= pressure_threshold:
                # failure: report failure at the index where pressure spike observed
                pick_type = 'Failed'
                print(f'Apple pick failed. Force: {filtered_force[0]}, dF/dt: {cropped_backward_diff}, Pressure: {avg_pressure}, index: {i}')
                return pick_type, i

        i += 1

    # if loop finishes without classification:
    return pick_type, pick_start if pick_start is not None else i

def picking_type_classifier_force_only(force, force_threshold, force_change_threshold):

    def moving_average(final_force):
        window_size = 5
        filtered = []
        for i in range(len(final_force) - window_size + 1):
            filtered.append(round(np.sum(final_force[i:i + window_size]) / window_size, 2))
        return filtered

    pick_type = 'Unclassified'
    i = 10

    while i < len(force) - 10:
        cropped_force = force[i - 10:i]
        filtered_force = moving_average(cropped_force)

        # Compute numeric derivative using your backwards-difference approach
        backwards_diff = []
        h = 2
        for j in range(2 * h, len(filtered_force)):
            diff = ((3 * filtered_force[j]) -
                    (4 * filtered_force[j - h]) +
                    (filtered_force[j - 2*h])) / (2 * h)
            backwards_diff.append(diff)

        avg_force_derivative = np.average(backwards_diff)

        # ---- Force-only logic ----
        # 1) Force must be low at baseline
        # 2) Force derivative spike → successful pick
        if filtered_force[0] <= force_threshold:
            if avg_force_derivative >= force_change_threshold:
                pick_type = 'Successful'
                print(f'Pick detected! dF/dt: {avg_force_derivative}')
                break

        i += 1

    # If never triggered
    if pick_type == 'Unclassified':
        print('Pick unclassified: no force spike detected.')

    return pick_type, i

def picking_type_classifier_f_tof(force, tof, tof_threshold, force_threshold, force_change_threshold):
    def moving_average(final_force):
        window_size = 5
        filtered = []
        for k in range(len(final_force) - window_size + 1):
            filtered.append(round(np.sum(final_force[k:k + window_size]) / window_size, 2))
        return filtered

    count = 0
    wait_for_tof_sync = 10     # same sync delay as pressure version
    pick_type = 'Unclassified'
    i = 10
    pick_start = None

    while i < len(force) - 10 and i < len(tof) - 10:   # make sure arrays don't overrun
        cropped_force = force[i - 10:i]
        filtered_force = moving_average(cropped_force)
        cropped_tof = tof[i - 10:i]
        avg_tof = np.average(cropped_tof)

        # backward (discrete) first derivative on filtered_force
        backwards_diff = []
        h = 2
        for j in range(2 * h, len(filtered_force)):
            diff = ((3 * filtered_force[j]) - (4 * filtered_force[j - h]) + filtered_force[j - 2*h]) / (2 * h)
            backwards_diff.append(diff)
        cropped_backward_diff = np.average(backwards_diff) if backwards_diff else 0.0

        if count > 0 and count < wait_for_tof_sync:
                    count += 1
        # picking logic starts here
        if filtered_force and filtered_force[0] <= force_threshold:
            # object detaches → force decreases fast, TOF small = object moved away cleanly
            if cropped_backward_diff >= force_change_threshold and avg_tof < tof_threshold:

                if pick_start is None:
                    pick_start = i  # record where pull-off detected

                if count < wait_for_tof_sync:
                    pick_type = 'Successful'   # assume Success unless TOF shows spike
                    i += 1
                    continue
                else:
                    print(f'Apple picked! dF/dt:{cropped_backward_diff}, TOF:{avg_tof}, detection_index:{pick_start}')
                    return 'Successful', pick_start

            # if TOF rises (or spikes) -> failed pick / stuck apple
            elif avg_tof >= tof_threshold and count < wait_for_tof_sync:
                print(f'Apple pick failed. Force:{filtered_force[0]}, dF/dt:{cropped_backward_diff}, TOF:{avg_tof}, index:{i}')
                return 'Failed', i
            elif count >= wait_for_tof_sync:
                return 'Successful', pick_start

        i += 1

    return pick_type, pick_start if pick_start is not None else i

def picking_type_classifier_force_flexnorm(
        force,
        flex_norm,
        flex_fail_threshold,      # e.g. 10 using your rule
        force_threshold,
        force_change_threshold):

    def moving_average(final_force):
        window_size = 5
        filtered = []
        for k in range(len(final_force) - window_size + 1):
            filtered.append(round(np.sum(final_force[k:k + window_size]) / window_size, 2))
        return filtered

    count = 0
    wait_for_flex_sync = 10
    pick_type = 'Unclassified'
    i = 10
    pick_start = None   # where spike first occurs

    while i < len(force) - 10:

        # 10-sample window
        cropped_force = force[i - 10:i]
        filtered_force = moving_average(cropped_force)

        cropped_flex = flex_norm[i - 10:i]
        avg_flex = np.average(cropped_flex)

        # backward derivative of filtered force
        backwards_diff = []
        h = 2
        for j in range(2 * h, len(filtered_force)):
            diff = ((3 * filtered_force[j])
                    - (4 * filtered_force[j - h])
                    + filtered_force[j - 2 * h]) / (2 * h)
            backwards_diff.append(diff)

        cropped_backward_diff = np.average(backwards_diff) if backwards_diff else 0.0

        if filtered_force and filtered_force[0] <= force_threshold:

            # CASE 1 — upward force spike AND flex has not collapsed
            if cropped_backward_diff >= force_change_threshold and avg_flex >= flex_fail_threshold:

                if pick_start is None:
                    pick_start = i

                # wait period to see if flex collapses afterwards
                if count < wait_for_flex_sync:
                    count += 1
                    pick_type = 'Successful'  # optimistic
                    i += 1
                    continue
                else:
                    # flex never collapsed → successful pick
                    pick_type = 'Successful'
                    print(f'Apple picked! dF/dt: {cropped_backward_diff}, FlexNorm: {avg_flex}, detection_index: {pick_start}')
                    return pick_type, pick_start

            # CASE 2 — flex norm dropped below threshold → FAILED PICK
            elif avg_flex < flex_fail_threshold:
                pick_type = 'Failed'
                print(f'Apple pick failed. Force: {filtered_force[0]}, '
                      f'dF/dt: {cropped_backward_diff}, FlexNorm: {avg_flex}, index: {i}')
                return pick_type, i

        i += 1

    # no decision reached
    return pick_type, (pick_start if pick_start is not None else i)

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
                    print(f"Apple picked! dF/dt:{dF}, sensors:{avg_values}, idx:{pick_start}")
                    return "Successful", pick_start

            # → Failed pick path
            elif failure_condition():
                # print(f"FAILED pick. dF/dt:{dF}, sensors:{avg_values}, idx:{i}")
                return "Failed", i

        i += 1

    return pick_type, pick_start if pick_start is not None else i

def picking_type_classifier_f_pressure_or_tof(force, pressure, tof,
                                              pressure_th, tof_th,
                                              force_th, force_change_th):
    sensors = {"pressure": pressure, "tof": tof}
    thresholds = {"pressure": pressure_th, "tof": tof_th}
    return picking_type_classifier_force_multi(
        force, sensors, thresholds, combination_logic="OR",
        force_threshold=force_th,
        force_change_threshold=force_change_th
    )

def picking_type_classifier_f_pressure_or_flex(force, pressure, flex,
                                               pressure_th, flex_th,
                                               force_th, force_change_th):
    sensors = {"pressure": pressure, "flex": -flex}
    thresholds = {"pressure": pressure_th, "flex": -flex_th}
    return picking_type_classifier_force_multi(
        force, sensors, thresholds, combination_logic="OR",
        force_threshold=force_th,
        force_change_threshold=force_change_th
    )

def picking_type_classifier_f_tof_or_flex(force, tof, flex,
                                          tof_th, flex_th,
                                          force_th, force_change_th):
    sensors = {"tof": tof, "flex": -flex}
    thresholds = {"tof": tof_th, "flex": -flex_th}
    return picking_type_classifier_force_multi(
        force, sensors, thresholds, combination_logic="OR",
        force_threshold=force_th,
        force_change_threshold=force_change_th
    )

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
    raw_f_arr, f_arr, etime_force = return_force_array(filename)
    p_arr, etimes_pressure = return_pressure_array(filename)
    tof_arr, etimes_tof = return_tof_array(filename)
    tof_arr_new = scipy.signal.resample(tof_arr, len(f_arr))
    flex_arr, etimes_flex = return_flex_array(filename)
    flex_arr_new = scipy.signal.resample(flex_arr, len(f_arr))
    p_arr_new = scipy.signal.resample(p_arr, len(f_arr))
    filtered_force = filter_force([f_arr], 21)[0]


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
    # pick_type, pick_i = picking_type_classifier_f_p(filtered_force[start_index:], p_arr_new[start_index:], 
                                                # pressure_threshold, force_threshold, force_change_threshold)
    # pick_type, pick_i = picking_type_classifier_force_only(filtered_force[start_index:], force_threshold, force_change_threshold)
    # pick_type, pick_i = picking_type_classifier_f_tof(filtered_force[start_index:-10], tof_arr_new[start_index:-10], pressure_threshold, force_threshold, force_change_threshold)
    # pick_type, pick_i = picking_type_classifier_force_flexnorm(filtered_force[start_index:-10], flex_arr_new[start_index:-10], pressure_threshold, force_threshold, force_change_threshold)
    
    # # calling force + (pressure or ToF)
    # pick_type, pick_i = picking_type_classifier_f_pressure_or_tof(
    #     filtered_force[start_index:-10],      # force
    #     p_arr_new[start_index:-10],           # pressure
    #     tof_arr_new[start_index:-10],         # tof
    #     pressure_threshold,                # pressure_th
    #     tof_threshold,                     # tof_th
    #     force_threshold,                   # force_th
    #     force_change_threshold             # force_change_th
    # )

    # # calling force + (pressure or flex)
    # pick_type, pick_i = picking_type_classifier_f_pressure_or_flex(
    #     filtered_force[start_index:-10],      # force
    #     p_arr_new[start_index:-10],           # pressure
    #     flex_arr_new[start_index:-10],         # flex
    #     pressure_threshold,                # pressure_th
    #     flex_threshold,                     # flex_th
    #     force_threshold,                   # force_th
    #     force_change_threshold             # force_change_th
    # )

    # # calling force + (tof or flex)
    # pick_type, pick_i = picking_type_classifier_f_tof_or_flex(
    #     filtered_force[start_index:-10],      # force
    #     tof_arr_new[start_index:-10],           # tof
    #     flex_arr_new[start_index:-10],         # flex
    #     tof_threshold,                # tof_th
    #     flex_threshold,                     # flex_th
    #     force_threshold,                   # force_th
    #     force_change_threshold             # force_change_th
    # )

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
    
    # plt.suptitle(f"Bag: {filename}\nPick Classification: {pick_type}\n(Actual Classification: Failed) ", fontsize=16)
    plt.suptitle(f"Bag: {filename}\nPick Classification: {pick_type}\n(Actual Classification: Successful) ", fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.95, 0.95])
    plt.show()

    print(f"\nBag: {filename}\nPick Classification: {pick_type}\n(Actual Classification: Successful) ")
    # print(f"\nBag: {filename}\nPick Classification: {pick_type}\n(Actual Classification: Failed) ")
