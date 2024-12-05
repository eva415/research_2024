import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pick_classification_functions import (make_confusion_matrix, process_file_and_graph_pick_analysis)


# Butterworth filter requirements
FS = 500.0  # sample rate, Hz
CUTOFF = 50  # desired cutoff frequency of the filter, Hz
ORDER = 2  # sin wave can be approx represented as quadratic

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
            print(f"Found directory: {file_path}")
            process_file_and_graph_pick_analysis(filename) # perform pick analysis and generate plot
            # uncomment to plot full data over time, no pick analysis
            # f_arr, etime_force = return_force_array(filename)
            # p_arr, etimes_pressure = return_pressure_array(filename)
            # total_disp, etime_joint = return_displacement_array(filename)
            #
            # # PLOT FIGS
            # fig, ax = plt.subplots(3, 1, figsize=(10, 30))
            # ax[0].plot(etime_force, f_arr)
            # ax[0].set_title(f'Norm(Force): /ft300_wrench\n/wrench/force/x, y, and z')
            # ax[0].set_xlabel('Displacement (m)')
            # ax[0].set_ylabel('Norm(Force) (N)')
            #
            # ax[1].plot(etimes_pressure, p_arr)  # etime = 42550	central_diff = 42450
            # ax[1].set_title(
            #     f'Pressure: /io_and_status_controller/io_states\n/analog_in_states[]/analog_in_states[1]/state')
            # ax[1].set_xlabel('Displacement (m)')
            # ax[1].set_ylabel('Pressure')
            #
            # ax[2].plot(etime_joint, total_disp)  # etime = 42550	central_diff = 42450
            # ax[2].set_title(f'Tool Pose (z-axis): /tool_pose\n/transform/translation/z')
            # ax[2].set_xlabel('Displacement (m)')
            # ax[2].set_ylabel('Z-position')
            #
            # plt.subplots_adjust(top=0.9, hspace=0.29)
            # fig.suptitle(
            #     f'file: {filename}')

            print(f"Finished directory: {file_path}\n")

    # make_confusion_matrix(positive_positive, negative_negative, positive_negative, negative_positive)

    p = PdfPages(pdf_title)  # initialize PDF object for plots
    fig_nums = plt.get_fignums()
    figs = [plt.figure(n) for n in fig_nums]  # iterating over the numbers in list
    for fig in figs:
        # and save/append that figure to the pdf file
        fig.savefig(p, format='pdf')
        plt.close(fig)

    # close the object to save the pdf
    p.close()


if __name__ == '__main__':
    bag_directory = "/home/evakrueger/Downloads/test_Rosbag_closed_loop"
    pdf_title = "pick_classifier_netherlands_nov25.pdf"

    loop_through_directory_save_plots(bag_directory, pdf_title)