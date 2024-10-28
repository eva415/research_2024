import matplotlib.pyplot as plt
import numpy as np
import scipy as scipy
from kneed import KneeLocator

from UR5e import UR5e_ros1
from ag_functions import bag_to_csv, butter_lowpass_filter, total_time, match_times, flipping_data, bag_pressure, \
    pressure_time, elapsed_time, filter_force, moving_average

FORCE_THRESHOLD_MAX = 5

FORCE_THRESHOLD_MIN = -1.0 # Defines min value for backward difference of the force that indicates valid picking action
# negative value = decrease in force over time

PRESSURE_THRESHOLD = 700


def picking_type_classifier(force_data, pressure):
    """
    Classifies the type of picking action based on force and pressure data.

    Parameters:
    force_data (list or numpy array): The force measurements during the picking process.
    pressure (list or numpy array): The pressure measurements during the picking process.

    Returns:
    tuple: A tuple containing:
        - pick_type (int): The result of the pick (1 = 'Successful' or 0 = 'Failed').
        - i (int): The index of the last evaluated force data point.
    """

    i = 10  # Initialize index for processing force data

    # Loop while i is greater than or equal to 10
    while i >= 10:
        idx = 0  # Initialize an index to track conditions
        cropped_force = force_data[i - 10:i]  # Crop the last 10 force readings for analysis
        filtered_force = moving_average(cropped_force)  # Apply moving average to the cropped force data
        cropped_pressure = pressure[i - 10:i]  # Crop the last 10 pressure readings for analysis
        avg_pressure = np.average(cropped_pressure)  # Calculate average pressure from the cropped data

        backwards_diff = []  # Initialize a list to store backward differences
        h = 2  # Set the step size for backward difference calculation

        # Calculate the backward difference for the filtered force data
        for j in range(2 * h, len(filtered_force)):
            diff = ((3 * filtered_force[j]) - (4 * filtered_force[j - h]) + filtered_force[j - (2 * h)]) / (2 * h)
            backwards_diff.append(diff)  # Append the calculated difference to the list

        # Calculate the average of the backward differences
        cropped_backward_diff = np.average(np.array(backwards_diff))

        # Check if the initial force reading is above the threshold
        if filtered_force[0] >= FORCE_THRESHOLD_MAX:
            # Condition for successful picking with low pressure
            if float(cropped_backward_diff) <= FORCE_THRESHOLD_MIN and avg_pressure < PRESSURE_THRESHOLD:
                pick_type = 1  # Mark as a successful pick
                # Uncomment the following line for debugging output
                # print(f'Apple has been picked! Bdiff: {cropped_backward_diff}   Pressure: {avg_pressure}. Force: {filtered_force[0]} vs. Max Force: {np.max(force)}')
                break  # Exit the loop

            # Condition for failed picking with high pressure
            elif float(cropped_backward_diff) <= FORCE_THRESHOLD_MAX and avg_pressure >= PRESSURE_THRESHOLD:
                pick_type = 0  # Mark as a failed pick
                # Uncomment the following line for debugging output
                # print(f'Apple was failed to be picked :( Force: {np.round(filtered_force[0])} Max Force: {np.max(force)}  Bdiff: {cropped_backward_diff}  Pressure: {avg_pressure}')
                break  # Exit the loop

            # If backward difference is not sufficient, continue checking
            elif float(cropped_backward_diff) > FORCE_THRESHOLD_MIN and np.round(filtered_force[0]) >= FORCE_THRESHOLD_MAX:
                idx += 1  # Increment the index for successful checks
                i += 1  # Move to the next force data point

        else:
            # If the initial force reading is below the threshold
            if idx == 0:
                i += 1  # Just increment index if no previous successful checks

            else:
                # Conditions for failed picking with low force readings
                if float(cropped_backward_diff) > FORCE_THRESHOLD_MIN and np.round(filtered_force[0]) < FORCE_THRESHOLD_MAX:
                    pick_type = 0  # Mark as a failed pick
                    # Uncomment the following line for debugging output
                    # print(f'Apple was failed to be picked :( Force: {np.round(filtered_force[0])} Max Force: {np.max(force)}  Bdiff: {cropped_backward_diff}  Pressure: {avg_pressure}')
                    break  # Exit the loop

                # Additional condition for failed picking due to pressure
                elif avg_pressure >= PRESSURE_THRESHOLD and np.round(filtered_force[0]) < FORCE_THRESHOLD_MAX:
                    pick_type = 0  # Mark as a failed pick
                    # Uncomment the following line for debugging output
                    # print(f'Apple was failed to be picked :( Force: {np.round(filtered_force[0])} Max Force: {np.max(force)}  Bdiff: {cropped_backward_diff}  Pressure: {avg_pressure}')
                    break  # Exit the loop

    return pick_type, i  # Return the type of pick and the last evaluated index


def plot_smoothed_displacement(time, dx, idx2, turn):
    plt.plot(time, dx)
    # plt.plot(p_dis, final_pressure)
    # plt.plot(p_dis[pturn1[0]], final_pressure[pturn1[0]], 'x')
    plt.plot(time[idx2 + 500], dx[idx2 + 500], "x")  # ax[2]
    plt.plot(time[turn[0]], dx[turn[0]], "x")
    plt.show()

def plot_force_displacement(new_x_part, cropped_f, cropped_low_bdiff, i, number, attempt, pick_type, general_time, actual):
    fig, ax = plt.subplots(2, 1, figsize=(10, 30))

    # Plot Norm(Force) vs. Distance Traveled
    ax[0].plot(new_x_part, cropped_f)
    ax[0].axvline(new_x_part[i], color='r')
    ax[0].set_title('Norm(Force) vs. Distance Traveled')
    ax[0].set_xlabel('Displacement (m)')
    ax[0].set_ylabel('Norm(Force) (N)')

    # Plot Numerical Derivative of Norm(Force) over Distance
    ax[1].plot(new_x_part, cropped_low_bdiff)
    ax[1].set_title('Numerical Derivative of Norm(Force) over Distance')
    ax[1].set_xlabel('Displacement (m)')
    ax[1].set_ylabel('Numerical Derivative of Norm(Force)')

    # Adjust layout and set figure title
    plt.subplots_adjust(top=0.9, hspace=0.29)
    fig.suptitle(
        f'Pick {number}-{attempt} Classification: {pick_type} Pick at Time {np.round(general_time[i], 2)} Seconds '
        f'(Actual Classification: {actual})'
    )

    plt.show()

def pick_analysis_from_file(filename, number, attempt):

    actual = f'Success'
    name = filename
    file = bag_to_csv(name)

    data = np.loadtxt('./' + file + '.csv', dtype="float", delimiter=',')

    # current guesstimation for x values - sample number
    sec = data[:, -1]
    nsec = data[:, -2]
    force = data[:, :-2]

    time = sec.tolist()
    ntime = nsec.tolist()

    # normalize data
    f_arr = np.linalg.norm(force, axis=1)

    tot_time = total_time(time, ntime)
    etime_force = elapsed_time(tot_time, tot_time)

    # displacement time baby
    path = UR5e_ros1(0, name)
    total_disp = path.z
    joint_times_sec = path.times_seconds
    joint_times_nsec = path.times_nseconds

    tot_time_j = total_time(joint_times_sec, joint_times_nsec)
    etime_joint = elapsed_time(tot_time_j, tot_time_j)

    pressure_array = bag_pressure(file)
    p_arr = np.linalg.norm(pressure_array, axis=1)

    etimes_pressure = pressure_time(file)

    f_arr_col = f_arr[..., None]
    total_disp_col = np.array(total_disp)[..., None]
    p_arr_col = p_arr[..., None]

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
    if number != 25:
        kn = KneeLocator(general_time, low_delta_x, curve='concave', direction='decreasing')
        kn1 = KneeLocator(general_time, low_delta_x, curve='convex', direction='decreasing')
        # idx1 = kn.minima_indices[0]
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

        cropped_x = delta_x[idx2 + 500:turn[0]]
        new_x_part = flipping_data(cropped_x)

    else:
        idx2 = np.where(low_delta_x == np.min(low_delta_x))[0][0]
        turn = np.where(low_delta_x == np.max(low_delta_x[idx2:-1]))[0]
        new_x_part = delta_x[idx2 + 500:turn[0]]

    # Plots displacement over time
    plot_smoothed_displacement(general_time, low_delta_x, idx2, turn)

    # cropped_x = delta_x[idx2 + 500:turn[0]]
    # new_x_part = flipping_data(cropped_x)
    # other_distance = new_distance(delta_x.tolist()[turn[0]+60:turn2[0]], new_x_part)
    # new_x = new_x_part + other_distance
    # final = reversed(fforce[peak:turn[0]])

    cropped_f = filtered[idx2 + 500:turn[0]]
    cropped_p = fp[idx2 + 500: turn[0]]
    flat_cropped_p = [element for innerList in cropped_p for element in innerList]
    cropped_low_bdiff = low_bdiff.tolist()[idx2 - 50 + 500:turn[0] - 50]
    flat_low_bdiff_o = [element for innerList in cropped_low_bdiff for element in innerList]

    pick_type, i = picking_type_classifier(cropped_f, cropped_p)
    plot_force_displacement(new_x_part, cropped_f, cropped_low_bdiff, i, number, attempt, pick_type, general_time, actual)
    return f'Pick {number}-{attempt} Predicted: {pick_type} Pick at Time {np.round(general_time[i], 2)} Seconds (Actual Pick: {actual})'

def main():
    number_test = 8
    attempt_test = 1
    filename_test = f'2023111_realapple{number_test}_mode_dual_attempt_{attempt_test}_orientation_0_yaw_0'  # 2023111_realapple20_mode_dual_attempt_2_orientation_0_yaw_0

    print(pick_analysis_from_file(filename_test, number_test, attempt_test))


if __name__ == "__main__":
    main()