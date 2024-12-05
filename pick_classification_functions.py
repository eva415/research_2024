from rclpy.serialization import deserialize_message
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
from scipy.signal import butter, filtfilt
from rosidl_runtime_py.utilities import get_message
import matplotlib.pyplot as plt
import numpy as np
import scipy as scipy
from kneed import KneeLocator
from scipy.ndimage import median_filter

INDEX_OG = 0 # originally, olivia had the index number set to 500

def elapsed_time(time_array):
    # Calculate the elapsed time relative to the first timestamp in an array
    # Convert to a NumPy array if it's not already (in case a list is passed)
    time_array = np.array(time_array)

    # Check if time_array is not empty
    if time_array.size == 0:
        print("Error: time_array is empty.")
        return None

    # Subtract the first entry from each element (vectorized operation)
    elapsed_t = time_array - time_array[0]

    # Optionally print for debugging
    # print(f"elapsed_time: {elapsed_t}")

    return elapsed_t
def total_time(seconds, nseconds):
    # Combine seconds and nanoseconds arrays into a single timestamp in seconds
    # Convert seconds and nanoseconds to total time in seconds using vectorized operations
    seconds = np.array(seconds, dtype=np.float64)
    nseconds = np.array(nseconds, dtype=np.float64)

    # Convert nanoseconds to seconds and add to seconds
    total = seconds + (nseconds / 1e9)

    # Optionally print for debugging
    # print(f"total_time: {total}")

    return total
def butter_lowpass_filter(data, cutoff, fs, order):
    # Apply a low-pass Butterworth filter to a data array
    nyq = 0.5 * fs  # Calculate Nyquist Frequency
    normal_cutoff = cutoff / nyq  # Normalize cutoff frequency
    # Get filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data, padlen=0)  # Apply the filter
    return y
def match_times(t1, t2, x1, x2):
    # Match timestamps and interpolate data arrays based on a common timestamp array
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
def filter_force(variables, param):
    # Applies a median filter to each variable --> noise reduction
    # Median Filter
    filtered = []
    for i in range(len(variables)):
        temp = median_filter(variables[i], param)
        filtered.append(temp)

    return filtered
def db3_to_csv_f(folder_name):
    # Reads a .db3 ros2 bag file and extracts relevant force data to a csv file
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
def db3_to_csv_p(folder_name):
    # Reads a .db3 ros2 bag file and extracts relevant pressure data to a csv file
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
                # P0 = msg.analog_in_states[0].state
                # P0 = P0*(-100.) + 1000.
                # print(f"P0: {P0}")
                P1 = msg.analog_in_states[1].state
                P1 = P1*(-100.) + 1000.
                # print(f"P1: {P1}")


                # Extract timestamp (seconds and nanoseconds) from message header
                secs = timestamp // 1_000_000_000  # Get the seconds part
                nsecs = timestamp % 1_000_000_000  # Get the nanoseconds part
                # print(f"secs: {secs}")
                # print(f"nsecs: {nsecs}\n")
                #
                # Compile extracted data into a list
                new_values = [P1, secs, nsecs]
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
def db3_to_csv_x(folder_name):
    # Reads a .db3 ros2 bag file and extracts relevant position data to a csv file
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
                # z_pos = msg.transform.translation.z

                # Extract timestamp (seconds and nanoseconds) from message header
                secs = msg.header.stamp.sec
                nsecs = msg.header.stamp.nanosec

                # Compile extracted data into a list
                new_values = [x_pos, y_pos, secs, nsecs]
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
def make_confusion_matrix(pos_pos, neg_neg, pos_neg, neg_pos):
    # Define the confusion matrix data
    matrix = np.array([[pos_pos, pos_neg],
                       [neg_pos, neg_neg]])

    # Define color map for each cell
    colors = [["green", "red"],
              ["red", "green"]]

    # Create the plot
    fig, ax = plt.subplots()
    ax.matshow([[1, 2], [3, 4]], cmap="Greys", alpha=0.1)  # Light background colors

    # Set tick locations to match the labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    # Set tick labels for each axis
    ax.set_xticklabels(['Positive', 'Negative'], fontsize=12)
    ax.set_yticklabels(['Positive', 'Negative'], fontsize=12)

    # Set x-axis and y-axis labels
    ax.set_xlabel('Predicted Label', fontsize=14)
    ax.set_ylabel('True Label', fontsize=14)

    # Place each value in the corresponding cell
    for (i, j), val in np.ndenumerate(matrix):
        # Choose color based on specified colors
        cell_color = "green" if colors[i][j] == "green" else "red"
        ax.text(j, i, f'{val}', ha='center', va='center', color=cell_color, fontsize=16)

    # Add gridlines for clearer separation
    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=2)

    plt.title("Confusion Matrix", fontsize=20)
    # plt.show()
def plot_array(array, time_array=None, xlabel="Time (s)", ylabel="Force", show=True):
    # If no time array is provided, use the indices of the force_array as time
    if time_array is None:
        time_array = np.arange(len(array))

    # Create the plot
    plt.plot(time_array, array, label=ylabel, color="b")

    # Add titles and labels
    plt.title(f"{ylabel} vs {xlabel}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Display a grid for better readability
    plt.grid(True)

    # Optionally display the legend
    plt.legend()

    # Show the plot
    if show:
        plt.show()
def return_force_array(filename):
    # FUNCTION TO RETRIEVE FORCE DATA, NORMALIZE
    # retrieve FORCE data from db3 file folder
    file_f = db3_to_csv_f(filename)
    data_f = np.loadtxt('./' + file_f + '.csv', dtype="float", delimiter=',')

    # get seconds and nanoseconds, process for total and elapsed time
    raw_sec_array_f = data_f[:, -2]
    raw_nsec_array_f = data_f[:, -1]
    total_time_force = total_time(raw_sec_array_f, raw_nsec_array_f)
    elapsed_time_force = elapsed_time(total_time_force)

    # get array of raw force data and normalize it
    raw_force_array = data_f[:, :-2]
    norm_force_array = np.linalg.norm(raw_force_array, axis=1)
    # plot_array(norm_force_array, time_array=elapsed_time_force, ylabel="Norm Force")
    return norm_force_array, elapsed_time_force
def return_displacement_array(filename):
    # retrieve POS data from db3 file folder
    file_pos = db3_to_csv_x(filename)
    data_pos = np.loadtxt('./' + file_pos + '.csv', dtype="float", delimiter=',')

    # get seconds and nanoseconds, process for total and elapsed time
    raw_sec_array_pos = data_pos[:, -2]
    raw_nsec_array_pos = data_pos[:, -1]
    total_time_pos = total_time(raw_sec_array_pos, raw_nsec_array_pos)
    elapsed_time_pos = elapsed_time(total_time_pos)

    # get array of raw force data and normalize it
    raw_pos_array = data_pos[:, :-2]
    norm_pos_array = np.linalg.norm(raw_pos_array, axis=1)
    # plot_array(raw_pos_array[:,0], time_array=elapsed_time_pos, ylabel="Total Displacement X")
    # plot_array(raw_pos_array[:,1], time_array=elapsed_time_pos, ylabel="Total Displacement Y")
    norm_pos_array = np.array(norm_pos_array).flatten()
    return norm_pos_array, elapsed_time_pos
def return_pressure_array(filename):
    # FUNCTION TO RETRIEVE PRESSURE DATA, NORMALIZE
    # retrieve PRESSURE data from db3 file folder
    file_p = db3_to_csv_p(filename)
    data_p = np.loadtxt('./' + file_p + '.csv', dtype="float", delimiter=',')

    # get seconds and nanoseconds, process for total and elapsed time
    raw_sec_array_p = data_p[:, -2]
    raw_nsec_array_p = data_p[:, -1]
    total_time_pressure = total_time(raw_sec_array_p, raw_nsec_array_p)
    elapsed_time_pressure = elapsed_time(total_time_pressure)

    # get array of raw force data and normalize it
    raw_pressure_array = data_p[:, :-2]
    # plot_array(raw_pressure_array, time_array=elapsed_time_pressure, ylabel="Pressure")
    return raw_pressure_array, elapsed_time_pressure
def picking_type_classifier(force, pressure, pressure_threshold, force_threshold, force_change_threshold):
    def moving_average(final_force):
        window_size = 5
        i = 0
        filtered = []

        while i < len(final_force) - window_size + 1:
            window_average = round(np.sum(final_force[i:i + window_size]) / window_size, 2)
            filtered.append(window_average)
            i += 1

        return filtered

    pick_type = None
    i = 10
    while i >= 10:
        idx = 0
        cropped_force = force[i - 10:i]
        filtered_force = moving_average(cropped_force)
        cropped_pressure = pressure[i - 10:i]
        avg_pressure = np.average(cropped_pressure)

        backwards_diff = []
        h = 2
        for j in range(2 * h, (len(filtered_force))):
            diff = ((3 * filtered_force[j]) - (4 * filtered_force[j - h]) + filtered_force[j - (2 * h)]) / (2 * h)
            backwards_diff.append(diff)
            j += 1

        cropped_backward_diff = np.average(np.array(backwards_diff))
        if filtered_force[0] >= force_threshold:
            if float(cropped_backward_diff) <= force_change_threshold and avg_pressure < pressure_threshold:  # this is the bitch to change if it stops working right
                pick_type = f'Successful'
                # print(f'Apple has been picked! Bdiff: {cropped_backward_diff}   Pressure: {avg_pressure}.\
                # Force: {filtered_force[0]} vs. Max Force: {np.max(force)}')
                break

            elif float(cropped_backward_diff) <= force_change_threshold and avg_pressure >= pressure_threshold:
                pick_type = f'Failed'
                # print(f'Apple was failed to be picked :( Force: {np.round(filtered_force[0])} Max Force: {np.max(force)}  Bdiff: {cropped_backward_diff}  Pressure: {avg_pressure}')
                break

            elif float(cropped_backward_diff) > force_change_threshold and np.round(filtered_force[0]) >= force_threshold:
                idx = idx + 1
                i = i + 1

        else:
            if idx == 0:
                i = i + 1

            else:
                if float(cropped_backward_diff) > force_change_threshold and np.round(filtered_force[0]) < force_threshold:
                    pick_type = f'Failed'
                    # print(f'Apple was failed to be picked :( Force: {np.round(filtered_force[0])} Max Force: {np.max(force)}  Bdiff: {cropped_backward_diff}  Pressure: {avg_pressure}')
                    break

                elif avg_pressure >= pressure_threshold and np.round(filtered_force[0]) < force_threshold:
                    pick_type = f'Failed'
                    # print(f'Apple was failed to be picked :( Force: {np.round(filtered_force[0])} Max Force: {np.max(force)}  Bdiff: {cropped_backward_diff}  Pressure: {avg_pressure}')
                    break

    return pick_type, i
def process_file_and_graph_pick_analysis(filename, pressure_threshold, force_threshold, force_change_threshold):
    flag = False  # reset every new pick analysis

    f_arr, etime_force = return_force_array(filename)
    p_arr, etimes_pressure = return_pressure_array(filename)
    total_disp, etime_joint = return_displacement_array(filename)

    # downsample to get same number of data points
    p_arr_new = scipy.signal.resample(p_arr, len(f_arr))

    # uncomment to view full plots of data over time
    # fig, ax = plt.subplots(3, 1, figsize=(10, 30))
    # ax[0].plot(etime_force, f_arr)
    # ax[0].set_title(f'Norm(Force): /ft300_wrench\n/wrench/force/x, y, and z')
    # ax[0].set_xlabel('Time (s)')
    # ax[0].set_ylabel('Norm(Force) (N)')
    #
    # ax[1].plot(etime_force, p_arr_new)  # etime = 42550	central_diff = 42450
    # ax[1].set_title(f'Pressure: /io_and_status_controller/io_states\n/analog_in_states[]/analog_in_states[1]/state')
    # ax[1].set_xlabel('Time (s)')
    # ax[1].set_ylabel('Pressure')
    #
    # ax[2].plot(etime_joint, total_disp)  # etime = 42550	central_diff = 42450
    # ax[2].set_title(f'Norm(Tool Pose): /tool_pose\n/transform/translation/x and y')
    # ax[2].set_xlabel('Time (s)')
    # ax[2].set_ylabel('Tool Pose')
    #
    # plt.subplots_adjust(top=0.9, hspace=0.29)
    # fig.suptitle(
    #     f'file: {FILENAME}')
    #
    # plt.show()

    f_arr_col = f_arr[..., None]
    total_disp_col = np.array(total_disp)[..., None]
    p_arr_col = p_arr_new

    final_force, delta_x, general_time = match_times(etime_force, etime_joint, f_arr_col, total_disp_col)
    final_pressure, p_dis, other_time = match_times(etime_force, etime_joint, p_arr_col, total_disp_col)

    fp = final_pressure.tolist()

    filtered = filter_force(final_force, 21)
    fp_new = scipy.signal.resample(fp, len(filtered)) # downsample to get right length array for indexing

    # Central Difference
    backwards_diff = []
    h = 2
    for j in range(2 * h, (len(filtered))):
        diff = ((3 * filtered[j]) - (4 * filtered[j - h]) + filtered[j - (2 * h)]) / (2 * h)
        backwards_diff.append(diff)
        j += 1

    # Filter requirements.
    fs = 500.0  # sample rate, Hz
    cutoff = 50  # desired cutoff frequency of the filter, Hz
    order = 2  # sin wave can be approx represented as quadratic

    low_bdiff = butter_lowpass_filter(backwards_diff, cutoff, fs, order)
    low_delta_x = scipy.signal.savgol_filter(delta_x[:, 0], 600, 1)

    # selecting the correct data to use
    kn1 = KneeLocator(general_time, low_delta_x, curve='convex', direction='decreasing')
    idx2 = kn1.minima_indices[0]
    idx2 = np.where(low_delta_x == np.max(low_delta_x[idx2:-1]))[0][0]
    turn = np.where(low_delta_x == np.min(low_delta_x[idx2:-1]))[0]
    # uncomment to see index check plot
    # plt.plot(general_time, low_delta_x)
    # plt.plot(general_time[idx2 + INDEX_OG], low_delta_x[idx2 + INDEX_OG], "x")  # ax[2]
    # plt.plot(general_time[turn[0]], low_delta_x[turn[0]], "x")
    # plt.show()
    cropped_f = filtered[idx2 + INDEX_OG:turn[0]]
    cropped_p = fp_new[idx2 + INDEX_OG: turn[0]]
    cropped_time = general_time[idx2 + INDEX_OG: turn[0]]
    cropped_low_bdiff = low_bdiff.tolist()[idx2 + INDEX_OG: turn[0]]

    # Moving average of dF/dt for plotting
    cropped_low_bdiff = np.asarray(cropped_low_bdiff).flatten()
    window_size = 5  # Choose an odd number, e.g., 5
    window = np.ones(window_size) / window_size
    moving_avg = np.convolve(cropped_low_bdiff, window, mode='same')

    pick_type, pick_i = picking_type_classifier(cropped_f, cropped_p, pressure_threshold, force_threshold, force_change_threshold)

    # plot of cropped force, cropped dF/dt, cropped pressure, and displacement
    fig, ax = plt.subplots(4, 1, figsize=(10, 40))

    ax[0].plot(general_time, low_delta_x)  # etime = 42550	central_diff = 42450
    ax[0].axvline(cropped_time[pick_i], color='r', linestyle='dotted')
    ax[0].fill_between(general_time, low_delta_x, 0,
                       where=(general_time >= cropped_time[0]) & (general_time <= cropped_time[-1]),
                       color='red', alpha=0.1)
    ax[0].set_title(f'FILTERED Norm(Tool Pose): /tool_pose\n/transform/translation/x and y')
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Tool Pose')

    ax[1].axvline(cropped_time[pick_i], color='r', linestyle='dotted')
    ax[1].plot(cropped_time, cropped_f)
    ax[1].set_title(f'CROPPED Norm(Force): /ft300_wrench\n/wrench/force/x, y, and z')
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('Norm(Force) (N)')

    ax[2].plot(cropped_time, moving_avg)
    ax[2].axvline(cropped_time[pick_i], color='r', linestyle='dotted')
    ax[2].set_title(f'MOVING AVERAGE CROPPED Norm(Force): /ft300_wrench\n/wrench/force/x, y, and z')
    ax[2].set_xlabel('Time (s)')
    ax[2].set_ylabel('Norm(Force) (N)')

    ax[3].plot(cropped_time, cropped_p)  # etime = 42550	central_diff = 42450
    ax[3].axvline(cropped_time[pick_i], color='r', linestyle='dotted')
    ax[3].set_title(
        f'CROPPED Pressure: /io_and_status_controller/io_states\n/analog_in_states[]/analog_in_states[1]/state')
    ax[3].set_xlabel('Time (s)')
    ax[3].set_ylabel('Pressure')


    # Adjust spacing
    plt.subplots_adjust(top=0.88, hspace=0.5)

    # Add a figure title with adjusted position
    fig.suptitle(
        f'{filename}\nPick Classification: {pick_type} Pick at Time {np.round(np.round(cropped_time[pick_i], 2), 2)} Seconds (Actual Classification: __)',
        y=0.95,
        fontsize=16
    )
    print(f'\tPick Classification: {pick_type} Pick at Time {np.round(np.round(cropped_time[pick_i], 2), 2)} Seconds (Actual Classification: __)')