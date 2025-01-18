import numpy as np
import rosbag
from geometry_msgs.msg import Wrench
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
import csv
from ag_functions import bag_to_csv, butter_lowpass_filter, total_time, match_times, flipping_data, bag_pressure, \
    pressure_time, elapsed_time, filter_force_m, db3_to_csv_f
from UR5e import UR5e_ros1
import scipy as scipy
from kneed import KneeLocator


INDEX = 500

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


number = 8
attempt = 1
actual = f'Success'
name = f'2023111_realapple{number}_mode_dual_attempt_{attempt}_orientation_0_yaw_0'  # 2023111_realapple20_mode_dual_attempt_2_orientation_0_yaw_0

file = db3_to_csv_f("pick_controller_20241023_115455.db3")

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
# path = UR5e_ros1(0, name)
# total_disp = path.z
# joint_times_sec = path.times_seconds
# joint_times_nsec = path.times_nseconds

# tot_time_j = total_time(joint_times_sec, joint_times_nsec)
# etime_joint = elapsed_time(tot_time_j, tot_time_j)

pressure_array = bag_pressure(file)
p_arr = np.linalg.norm(pressure_array, axis=1)

etimes_pressure = pressure_time(file)

f_arr_col = f_arr[..., None]
# total_disp_col = np.array(total_disp)[..., None]
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

# selecteding the correct data to use
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

    cropped_x = delta_x[idx2 + INDEX:turn[0]]
    new_x_part = flipping_data(cropped_x)

else:
    idx2 = np.where(low_delta_x == np.min(low_delta_x))[0][0]
    turn = np.where(low_delta_x == np.max(low_delta_x[idx2:-1]))[0]
    new_x_part = delta_x[idx2 + INDEX:turn[0]]

plt.plot(general_time, low_delta_x)
# plt.plot(p_dis, final_pressure)
# plt.plot(p_dis[pturn1[0]], final_pressure[pturn1[0]], 'x')
plt.plot(general_time[idx2 + INDEX], low_delta_x[idx2 + INDEX], "x")  # ax[2]
plt.plot(general_time[turn[0]], low_delta_x[turn[0]], "x")
plt.show()

# cropped_x = delta_x[idx2 + 500:turn[0]]
# new_x_part = flipping_data(cropped_x)
# other_distance = new_distance(delta_x.tolist()[turn[0]+60:turn2[0]], new_x_part)
# new_x = new_x_part + other_distance
# final = reversed(fforce[peak:turn[0]])

cropped_f = filtered[idx2 + INDEX:turn[0]]
cropped_p = fp[idx2 + INDEX: turn[0]]
flat_cropped_p = [element for innerList in cropped_p for element in innerList]
cropped_low_bdiff = low_bdiff.tolist()[idx2 - 50 + INDEX:turn[0] - 50]
flat_low_bdiff_o = [element for innerList in cropped_low_bdiff for element in innerList]

type, i = picking_type_classifier(cropped_f, cropped_p)

fig, ax = plt.subplots(2, 1, figsize=(10, 30))
ax[0].plot(new_x_part, cropped_f)
ax[0].axvline(new_x_part[i], color='r')
ax[0].set_title('Norm(Force) vs. Distance Traveled')
ax[0].set_xlabel('Displacement (m)')
ax[0].set_ylabel('Norm(Force) (N)')

ax[1].plot(new_x_part, cropped_low_bdiff)  # etime = 42550    central_diff = 42450
ax[1].set_title('Numerical Derivative of Norm(Force) over Distance')
ax[1].set_xlabel('Displacement (m)')
ax[1].set_ylabel('Numerical Derivative of Norm(Force)')

plt.subplots_adjust(top=0.9, hspace=0.29)
fig.suptitle(
    f'Pick {number}-{attempt} Classification: {type} Pick at Time {np.round(general_time[i], 2)} Seconds (Actual Classification: {actual})')

plt.show()