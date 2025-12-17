import os
from wsgiref.validate import bad_header_value_re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from rf_pick_classification_funcs import (process_file_and_graph_pick_analysis, return_force_array, return_pressure_array)

# values to change to improve pick analysis:
engaged_pressure = -56
# disengaged_pressure = -50
failure_ratio = 0.57
PRESSURE_THRESHOLD =  -56
FORCE_CHANGE_THRESHOLD = 1.0
FORCE_THRESHOLD = 20
TOF_THRESHOLD = 55
FLEX_THRESHOLD = 20

# ANSI escape codes for colors
RESET = "\033[0m"
RED = "\033[91m"


def loop_through_directory_save_plots(directory_path, pdf_title):
    # # information for confusion matrix RIGHT NOW EXAMPLE VALUES
    # total_count_attempts = 1
    # total_correct_classifications = 1
    # positive_positive = 1  # positive means successful pick
    # negative_negative = 0  # negative means failed pick
    # positive_negative = 0  # position classification, negative actual
    # negative_positive = 0  # negative classification, positive actual

    os.chdir(directory_path)

    # Loop through all files in the directory
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        # Check if it's a file (not a directory)
        if os.path.isdir(file_path):
            # print(f"Found directory: {file_path}")
            try:
                process_file_and_graph_pick_analysis(filename, PRESSURE_THRESHOLD, FORCE_THRESHOLD, FORCE_CHANGE_THRESHOLD, TOF_THRESHOLD, FLEX_THRESHOLD) # perform pick analysis and generate plot
            except Exception as e:
                print(f"\tERROR: {e}")
            # # uncomment to plot full data over time, no pick analysis
            # f_arr, fz_arr, etime_force = return_force_array(filename)
            # p_arr, etimes_pressure = return_pressure_array(filename)
            # total_disp, etime_joint = return_displacement_array(filename)
            
            # # PLOT FIGS
            # fig, ax = plt.subplots(3, 1, figsize=(10, 30))
            # norm_f_arr = np.linalg.norm(f_arr, axis=1)
            # ax[0].plot(etime_force, norm_f_arr)
            # ax[0].set_title(f'Norm(Force): /ft300_wrench\n/wrench/force/x, y, and z')
            # ax[0].set_xlabel('Displacement (m)')
            # ax[0].set_ylabel('Norm(Force) (N)')
            
            # ax[1].plot(etimes_pressure, p_arr)  # etime = 42550	central_diff = 42450
            # ax[1].set_title(
            #     f'Pressure: /io_and_status_controller/io_states\n/analog_in_states[]/analog_in_states[1]/state')
            # ax[1].set_xlabel('Displacement (m)')
            # ax[1].set_ylabel('Pressure')
            
            # ax[2].plot(etime_joint, total_disp)  # etime = 42550	central_diff = 42450
            # ax[2].set_title(f'Tool Pose (z-axis): /tool_pose\n/transform/translation/z')
            # ax[2].set_xlabel('Displacement (m)')
            # ax[2].set_ylabel('Z-position')
            
            # plt.subplots_adjust(top=0.9, hspace=0.29)
            # fig.suptitle(
            #     f'file: {filename}')

            # print(f"Finished directory: {file_path}\n")

    p = PdfPages(pdf_title)  # initialize PDF object for plots
    fig_nums = plt.get_fignums()
    figs = [plt.figure(n) for n in fig_nums]  # iterating over the numbers in list
    for fig in figs:
        # and save/append that figure to the pdf file
        fig.savefig(p, format='pdf')
        plt.close(fig)

    # close the object to save the pdf
    p.close()


def detect_outliers(data, threshold=20):
    outlier_indices = np.where(np.abs(data) > threshold)[0]
    return outlier_indices
def remove_outliers(data, outlier_indices):
    cleaned_data = data.copy()
    for idx in outlier_indices:
        # Check bounds to ensure we can take the previous and next samples
        if idx > 0 and idx < len(data) - 1:
            cleaned_data[idx] = (data[idx - 1] + data[idx + 1]) / 2
        elif idx == 0:  # Handle edge case at the start
            cleaned_data[idx] = data[idx + 1]
        elif idx == len(data) - 1:  # Handle edge case at the end
            cleaned_data[idx] = data[idx - 1]
    return cleaned_data
# graph for figure 6:
def graph_all_forces(directory_path, pdf_title):
    os.chdir(directory_path)

    # Loop through all files in the directory
    all_fx = []
    all_fy = []
    all_fz = []
    fig, ax = plt.subplots(figsize=(10, 10))
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        # Check if it's a file (not a directory)
        if os.path.isdir(file_path):
            print(f"Found directory: {file_path}")
            try:
                raw_f_arr, f_arr, etime_force = return_force_array(filename)

                x_outliers = detect_outliers(raw_f_arr[:, 0])
                x_forces_filtered = remove_outliers(raw_f_arr[:, 0], x_outliers)
                all_fx.append(max(abs(x_forces_filtered)))

                y_outliers = detect_outliers(raw_f_arr[:, 1])
                y_forces_filtered = remove_outliers(raw_f_arr[:, 1], y_outliers)
                all_fy.append(max(abs(y_forces_filtered)))

                z_forces = raw_f_arr[:, 2]
                all_fz.append(max(abs(z_forces)))

            except Exception as e:
                print(f"\tERROR: {e}")
            # print(np.arange(len(all_fx)))
            plt.tick_params(axis='x', labelsize=30)  # For x-axis of the first subplot
            plt.tick_params(axis='y', labelsize=30)  # For y-axis of the first subplot
            plt.xlim(0, 105)
            plt.bar(np.arange(len(all_fx)), all_fx, color='#ff7f00')
            plt.bar(np.arange(len(all_fy))+35, all_fy, color='#4daf4a')
            plt.bar(np.arange(len(all_fz))+70, all_fz, color='#984ea3')
            plt.text(10, -5, r"$F_x$", ha='center', fontsize=30)
            plt.text(50, -5, r"$F_y$", ha='center', fontsize=30)
            plt.text(90, -5, r"$F_z$", ha='center', fontsize=30)

            plt.ylabel('Max Force (N)', fontsize=35)

            print(f"Finished directory: {file_path}\n")


    p = PdfPages(pdf_title)  # initialize PDF object for plots
    fig_nums = plt.get_fignums()
    figs = [plt.figure(n) for n in fig_nums]  # iterating over the numbers in list
    for fig in figs:
        # and save/append that figure to the pdf file
        fig.savefig(p, format='pdf')
        plt.close(fig)

    # close the object to save the pdf
    p.close()
    return None

def parallel_line_plot(directory_path, pdf_title):
    os.chdir(directory_path)
    # Generate some sample data
    user_pick_time = np.array([
    44.44, 58.9, 38.22, 47.77, 59.31, 73.52, 44.59, 47.9, 37.32, 45.88,
    41.97, 63.63, 46.19, 51.78, 38.33, 41.26, 50.4, 24.13, 37.01,
    25.44, 25.37, 26.6, 28.51, 25.21, 46.41, 30.48, 26.51, 27.08
])  # Actual pick time
    estimated_pick_time = np.array([
    44.13, 58.65, 37.86, 47.11, 58.98, 73.11, 44.18, 47.48, 36.74, 44.75,
    41.72, 61.38, 44.7, 51.26, 37.86, 40.02, 48.29, 23.62, 36.61, 25.02,
    24.82, 26.21, 27.88, 24.98, 45.98, 30.06, 25.92, 26.63
])  # Estimated pick time

    # Stack the paired data into a 2D array where each column represents one dataset
    data = np.stack((user_pick_time, estimated_pick_time), axis=1)

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(12, 12))

    plt.tick_params(axis='x', labelsize=30)  # For x-axis of the first subplot
    # Plot the paired data as dots connected by lines
    for i in range(len(user_pick_time)):
        ax.plot([user_pick_time[i], estimated_pick_time[i]], [i, i], 'k-', lw=1)  # Line between the two dots
        ax.plot(user_pick_time[i], i, 'o', markersize=10, color="#4daf4a", label=f'Pick time' if i == 0 else "")  # dot for pick time
        ax.plot(estimated_pick_time[i], i, 'o', markersize=10, color="#984ea3",
                label=f'Estimated time' if i == 0 else "")  # dot for estimated pick time

    # Set labels and title
    ax.set_xlabel('Time', fontsize=35)
    ax.set_title('User Pick Time vs Classifier Pick Time', fontsize=35)
    ax.yaxis.set_visible(False)
    ax.legend(fontsize=30)

    p = PdfPages(pdf_title)  # initialize PDF object for plots
    fig_nums = plt.get_fignums()
    figs = [plt.figure(n) for n in fig_nums]  # iterating over the numbers in list
    for fig in figs:
        # and save/append that figure to the pdf file
        fig.savefig(p, format='pdf')
        plt.close(fig)

    # close the object to save the pdf
    p.close()
    return None

if __name__ == '__main__':
    # bag_directory = "/home/imml/Desktop/successful_picks"
    bag_directory = "/home/imml/Desktop/failed_picks"
    pdf_title = "fail_test.pdf"
    # pdf_title = "success_test.pdf"

    loop_through_directory_save_plots(bag_directory, pdf_title)
    # graph_all_forces(bag_directory, pdf_title)
    # parallel_line_plot(bag_directory, pdf_title)