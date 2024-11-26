import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from ag_functions import (match_times, moving_average, db3_to_csv_f, db3_to_csv_p, db3_to_csv_x, total_time,
                          elapsed_time, filter_force, butter_lowpass_filter, flipping_data, pandas_to_merge_timestamps)
import scipy as scipy
from kneed import KneeLocator


# Butterworth filter requirements
FS = 500.0  # sample rate, Hz
CUTOFF = 50  # desired cutoff frequency of the filter, Hz
ORDER = 2  # sin wave can be approx represented as quadratic

# ANSI escape codes for colors
RESET = "\033[0m"
RED = "\033[91m"

# Directory of the bag files
DIRECTORY = "/home/evakrueger/Downloads/Rosbag_closed_loop"
FILENAME = "rosbag2_2024_09_20-10_20_11_0.db3"
# PRESSURE_THRESHOLD = 700 # this is the thing to change if it stops working right: originally was 700
engaged_pressure = 550.0
disengaged_pressure = 1000.0
failure_ratio = 0.57
PRESSURE_THRESHOLD = engaged_pressure + failure_ratio * (disengaged_pressure - engaged_pressure) # 806.5
FORCE_CHANGE_THRESHOLD = -1.0
FORCE_THRESHOLD = 5
INDEX_OG = 500 # originally, olivia had the index number set to 500
PDF_TITLE = "pick_classifier_netherlands_nov25.pdf"


def wur_event_detect(force_array, pressure_array, active = 0, window = 10, flag = False):
    # if for some reason the engaged pressure is higher, flip > and < for pressure

    if len(force_array) < window or len(pressure_array) < window:
        return

    if active == 0:
        return

    filtered_force = moving_average(force_array)
    avg_pressure = np.average(pressure_array)

    backwards_diff = []
    h = 2
    for j in range(2 * h, (len(filtered_force))):
        diff = ((3 * filtered_force[j]) - (4 * filtered_force[j - h]) + filtered_force[j - (2 * h)]) / (2 * h)
        backwards_diff.append(diff)
        j += 1

    cropped_backward_diff = np.average(np.array(backwards_diff))

    # if the suction cups are disengaged, the pick failed
    if avg_pressure >= PRESSURE_THRESHOLD:
        print(
            f"Apple was failed to be picked :( Force: {np.round(filtered_force[0])} Max Force: {np.max(filtered_force)}  Bdiff: {cropped_backward_diff}  Pressure: {avg_pressure}")
        return 0

    # if there is a reasonable force
    elif filtered_force[0] >= 5:

        flag = True  # force was achieved

        # check for big force drop
        if float(cropped_backward_diff) <= FORCE_CHANGE_THRESHOLD and avg_pressure < PRESSURE_THRESHOLD:
            print(f"Apple has been picked! Bdiff: {cropped_backward_diff}   Pressure: {avg_pressure}.\
                    #Force: {filtered_force[0]} vs. Max Force: {np.max(force_array)}")
            return 1

        elif float(
                cropped_backward_diff) <= FORCE_CHANGE_THRESHOLD and avg_pressure >= PRESSURE_THRESHOLD:
            print(
                f"Apple was failed to be picked :( Force: {np.round(filtered_force[0])} Max Force: {np.max(force_array)}  Bdiff: {cropped_backward_diff}  Pressure: {avg_pressure}")
            return 0

    # if force is low, but was high, that's a failure too
    elif flag and filtered_force[0] < 4.5:
        print(
            f"Apple was failed to be picked :( Force: {np.round(filtered_force[0])} Max Force: {np.max(force_array)}  Bdiff: {cropped_backward_diff}  Pressure: {avg_pressure}")
        return 0
# YAY Given force and pressure data, determine if pick occurred (cb)
def pick_analysis_callback(force_data, pressure_data, flag=False):
    if len(force_data) < 9:
        print(f"Force array is of length {len(force_data)}")
        return 2

    # Smooth the force data with a moving average filter
    filtered_force = moving_average(force_data, window_size=5)
    avg_pressure = np.average(pressure_data)

    # Central difference computation for the derivative of force
    backwards_diff = []
    h = 2  # Smoothing parameter for central difference

    for j in range(2 * h, len(filtered_force)):
        diff = ((3 * filtered_force[j]) - (4 * filtered_force[j - h]) + filtered_force[j - (2 * h)]) / (2 * h)
        backwards_diff.append(diff)

    # Low-pass filter the backward differences (derivative of force)
    fs = 500.0  # Sample rate in Hz (adjust as necessary)
    cutoff = 50  # Desired cutoff frequency of the low-pass filter in Hz (adjust as necessary)
    low_bdiff = butter_lowpass_filter(backwards_diff, cutoff, fs, order=2)

    # Compute the average of the filtered backward differences
    cropped_backward_diff = np.average(np.array(low_bdiff))

    # if the suction cups are disengaged, the pick failed
    if avg_pressure >= PRESSURE_THRESHOLD:
        print(
            f"Apple was failed to be picked :( Force: {np.round(filtered_force[0])} Max Force: {np.max(filtered_force)} Bdiff: {cropped_backward_diff} Pressure: {avg_pressure}")
        return 0

    # if there is a reasonable force
    elif filtered_force[0] >= 5:
        flag = True  # force was achieved
        # check for big force drop
        if cropped_backward_diff <= FORCE_THRESHOLD and avg_pressure < PRESSURE_THRESHOLD:
            print(
                f"Apple has been picked! Bdiff: {cropped_backward_diff} Pressure: {avg_pressure}. Force: {filtered_force[0]} vs. Max Force: {np.max(force_data)}")
            return 1

        elif cropped_backward_diff <= FORCE_THRESHOLD and avg_pressure >= PRESSURE_THRESHOLD:
            print(
                f"Apple was failed to be picked :( Force: {np.round(filtered_force[0])} Max Force: {np.max(force_data)} Bdiff: {cropped_backward_diff} Pressure: {avg_pressure}")
            return 0
        else:
            print("Failed to identify pick type... Force was high and idk man")
            return 2

    # if force is low, but was high, that's a failure too
    elif flag and filtered_force[0] < 4.5:
        print(
            f"Apple was failed to be picked :( Force: {np.round(filtered_force[0])} Max Force: {np.max(force_data)} Bdiff: {cropped_backward_diff} Pressure: {avg_pressure}")
        return 0

    else:
        print("Failed to identify pick type... Force was high and remains high")
        return 2
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
def DEAL_WITH_THIS_LATER():
    # information for confusion matrix RIGHT NOW EXAMPLE VALUES
    total_count_attempts = 1
    total_correct_classifications = 1
    positive_positive = 1  # positive means successful pick
    negative_negative = 0  # negative means failed pick
    positive_negative = 0  # position classification, negative actual
    negative_positive = 0  # negative classification, positive actual
    make_confusion_matrix(positive_positive, negative_negative, positive_negative, negative_positive)

    p = PdfPages(PDF_TITLE)
    fig_nums = plt.get_fignums()

    figs = [plt.figure(n) for n in fig_nums]  # iterating over the numbers in list

    for fig in figs:
        # and save/append that figure to the pdf file
        fig.savefig(p, format='pdf')
        plt.close(fig)

    # close the object to save the pdf
    p.close()
def plot_array(array, time_array=None, xlabel="Time (s)", ylabel="Force", show=False):
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
    # print(f"elapsed_time_force: {elapsed_time_force}")

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
    # norm_pos_array = np.linalg.norm(raw_pos_array, axis=1)
    # plot_array(raw_pos_array, time_array=elapsed_time_pos, ylabel="Total Displacement")
    raw_pos_array = np.array(raw_pos_array).flatten()
    return raw_pos_array, elapsed_time_pos
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
def picking_type_classifier(force, pressure):
    def moving_average(final_force):
        window_size = 5
        i = 0
        filtered = []

        while i < len(final_force) - window_size + 1:
            window_average = round(np.sum(final_force[i:i + window_size]) / window_size, 2)
            filtered.append(window_average)
            i += 1

        return filtered

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

        if filtered_force[0] >= 5:
            if float(
                    cropped_backward_diff) <= -1.0 and avg_pressure < 700:  # this is the bitch to change if it stops working right
                type = f'Successful'
                # print(f'Apple has been picked! Bdiff: {cropped_backward_diff}   Pressure: {avg_pressure}.\
                # Force: {filtered_force[0]} vs. Max Force: {np.max(force)}')
                break

            elif float(cropped_backward_diff) <= -1.0 and avg_pressure >= 700:
                type = f'Failed'
                # print(f'Apple was failed to be picked :( Force: {np.round(filtered_force[0])} Max Force: {np.max(force)}  Bdiff: {cropped_backward_diff}  Pressure: {avg_pressure}')
                break

            elif float(cropped_backward_diff) > -1.0 and np.round(filtered_force[0]) >= 5:
                idx = idx + 1
                i = i + 1

        else:
            if idx == 0:
                i = i + 1

            else:
                if float(cropped_backward_diff) > -1.0 and np.round(filtered_force[0]) < 5:
                    type = f'Failed'
                    # print(f'Apple was failed to be picked :( Force: {np.round(filtered_force[0])} Max Force: {np.max(force)}  Bdiff: {cropped_backward_diff}  Pressure: {avg_pressure}')
                    break

                elif avg_pressure >= 700 and np.round(filtered_force[0]) < 5:
                    type = f'Failed'
                    # print(f'Apple was failed to be picked :( Force: {np.round(filtered_force[0])} Max Force: {np.max(force)}  Bdiff: {cropped_backward_diff}  Pressure: {avg_pressure}')
                    break

    return type, i
def new_try():
    flag = False  # reset every new pick analysis
    os.chdir(DIRECTORY) # go to desired directory with bag files

    f_arr, etime_force = return_force_array(FILENAME)
    p_arr, etimes_pressure = return_pressure_array(FILENAME)
    total_disp, etime_joint = return_displacement_array(FILENAME)


    fig, ax = plt.subplots(3, 1, figsize=(10, 30))
    ax[0].plot(etime_force, f_arr)
    ax[0].set_title(f'Norm(Force): /ft300_wrench\n/wrench/force/x, y, and z')
    ax[0].set_xlabel('Displacement (m)')
    ax[0].set_ylabel('Norm(Force) (N)')

    ax[1].plot(etimes_pressure, p_arr)  # etime = 42550	central_diff = 42450
    ax[1].set_title(f'Pressure: /io_and_status_controller/io_states\n/analog_in_states[]/analog_in_states[1]/state')
    ax[1].set_xlabel('Displacement (m)')
    ax[1].set_ylabel('Pressure')

    ax[2].plot(etime_joint, total_disp)  # etime = 42550	central_diff = 42450
    ax[2].set_title(f'Tool Pose (z-axis): /tool_pose\n/transform/translation/z')
    ax[2].set_xlabel('Displacement (m)')
    ax[2].set_ylabel('Z-position')

    plt.subplots_adjust(top=0.9, hspace=0.29)
    fig.suptitle(
        f'file: {FILENAME}')

    plt.show()

    f_arr_col = f_arr[..., None]
    total_disp_col = np.array(total_disp)[..., None]
    p_arr_col = p_arr
    final_force, delta_x, general_time = match_times(etime_force, etime_joint, f_arr_col, total_disp_col)

    final_pressure, p_dis, other_time = match_times(etimes_pressure, etime_joint, p_arr_col, total_disp_col)

    fp = final_pressure.tolist()

    filtered = filter_force(final_force, 21)

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

    kn = KneeLocator(general_time, low_delta_x, curve='concave', direction='decreasing')
    kn1 = KneeLocator(general_time, low_delta_x, curve='convex', direction='decreasing')
    idx2 = kn1.minima_indices[0]
    idx = kn1.minima_indices[-2]
    peaks, peak_index = scipy.signal.find_peaks(low_delta_x, height=0.03, plateau_size=(None, None))
    peak = int(peak_index['right_edges'][-1])
    peak1 = int(peak_index['left_edges'][0])

    # pressure turn
    pturn1 = np.where(final_pressure == np.min(final_pressure))[0]

    # turn = [i for i, e in enumerate(low_delta_x) if e < (low_delta_x[peak]-0.16) and i > peak]
    idx2 = np.where(low_delta_x == np.max(low_delta_x[idx2:-1]))[0][0]
    turn = np.where(low_delta_x == np.min(low_delta_x[idx2:-1]))[
        0]  # this is not picking where the turn around happens exactly
    turn2 = np.where(low_delta_x == np.max(low_delta_x[idx2:-1]))[0]

    cropped_x = delta_x[idx2 + INDEX_OG:turn[0]]
    new_x_part = flipping_data(cropped_x)

    plt.plot(general_time, low_delta_x)
    plt.plot(general_time[idx2 + INDEX_OG], low_delta_x[idx2 + INDEX_OG], "x")  # ax[2]
    plt.plot(general_time[turn[0]], low_delta_x[turn[0]], "x")
    plt.show()

    cropped_f = filtered[idx2 + INDEX_OG:turn[0]]
    cropped_p = fp[idx2 + INDEX_OG: turn[0]]
    cropped_low_bdiff = low_bdiff.tolist()[idx2 - 50 + INDEX_OG:turn[0] - 50]

    type, i = picking_type_classifier(cropped_f, cropped_p)
    k = 0
    while (k+15 < len(cropped_f)) and (k+15 < len(cropped_p)):
        result = wur_event_detect(f_arr[k:k + 15], p_arr[k:k + 15])
        print(f"k: {k}, result: {result}")
        k += 15

    fig, ax = plt.subplots(2, 1, figsize=(10, 30))
    ax[0].plot(new_x_part, cropped_f)
    ax[0].axvline(new_x_part[i], color='r')
    ax[0].set_title('Norm(Force) vs. Distance Traveled')
    ax[0].set_xlabel('Displacement (m)')
    ax[0].set_ylabel('Norm(Force) (N)')

    ax[1].plot(new_x_part, cropped_low_bdiff)  # etime = 42550	central_diff = 42450
    ax[1].set_title('Numerical Derivative of Norm(Force) over Distance')
    ax[1].set_xlabel('Displacement (m)')
    ax[1].set_ylabel('Numerical Derivative of Norm(Force)')

    plt.subplots_adjust(top=0.9, hspace=0.29)
    fig.suptitle(
        f'Pick Classification: {type} Pick at Time {np.round(general_time[i], 2)} Seconds (Actual Classification: __)')

    plt.show()
def loop_through_directory_save_plots(directory_path=DIRECTORY):
    os.chdir(DIRECTORY)

    # Loop through all files in the directory
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        # Check if it's a file (not a directory)
        if os.path.isdir(file_path):
            print(f"Found directory: {file_path}")
            f_arr, etime_force = return_force_array(filename)
            p_arr, etimes_pressure = return_pressure_array(filename)
            total_disp, etime_joint = return_displacement_array(filename)

            # PLOT FIGS
            fig, ax = plt.subplots(3, 1, figsize=(10, 30))
            ax[0].plot(etime_force, f_arr)
            ax[0].set_title(f'Norm(Force): /ft300_wrench\n/wrench/force/x, y, and z')
            ax[0].set_xlabel('Displacement (m)')
            ax[0].set_ylabel('Norm(Force) (N)')

            ax[1].plot(etimes_pressure, p_arr)  # etime = 42550	central_diff = 42450
            ax[1].set_title(
                f'Pressure: /io_and_status_controller/io_states\n/analog_in_states[]/analog_in_states[1]/state')
            ax[1].set_xlabel('Displacement (m)')
            ax[1].set_ylabel('Pressure')

            ax[2].plot(etime_joint, total_disp)  # etime = 42550	central_diff = 42450
            ax[2].set_title(f'Tool Pose (z-axis): /tool_pose\n/transform/translation/z')
            ax[2].set_xlabel('Displacement (m)')
            ax[2].set_ylabel('Z-position')

            plt.subplots_adjust(top=0.9, hspace=0.29)
            fig.suptitle(
                f'file: {FILENAME}')

            print(f"Finished directory: {file_path}\n")

    p = PdfPages(PDF_TITLE)  # initialize PDF object for plots
    fig_nums = plt.get_fignums()
    figs = [plt.figure(n) for n in fig_nums]  # iterating over the numbers in list
    for fig in figs:
        # and save/append that figure to the pdf file
        fig.savefig(p, format='pdf')
        plt.close(fig)
    # close the object to save the pdf
    p.close()


if __name__ == '__main__':
    loop_through_directory_save_plots()