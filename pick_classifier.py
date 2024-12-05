from pick_classification_functions import (loop_through_directory_save_plots)


# Butterworth filter requirements
FS = 500.0  # sample rate, Hz
CUTOFF = 50  # desired cutoff frequency of the filter, Hz
ORDER = 2  # sin wave can be approx represented as quadratic

# ANSI escape codes for colors
RESET = "\033[0m"
RED = "\033[91m"


if __name__ == '__main__':
    bag_directory = "/home/evakrueger/Downloads/test_Rosbag_closed_loop"
    pdf_title = "pick_classifier_netherlands_nov25.pdf"

    loop_through_directory_save_plots(bag_directory, pdf_title)