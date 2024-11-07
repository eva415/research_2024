
# Olivia's function - going to use pick_analysis_callback instead
def picking_type_classifier(force, pressure):
    i = 10
    # print("INSIDE PICKING TYPE CLASSIFIER FUNCTION")
    # print("force length: ", len(force))
    # print("pressure length: ", len(pressure))
    while i >= 10:
        idx = 0
        cropped_force = force[i - 10:i]
        # print("cropped force length: ", len(cropped_force))
        filtered_force = moving_average(cropped_force)
        # print("filtered force length: ", len(filtered_force))
        cropped_pressure = pressure[i - 10:i]
        # print(f"cropped pressure length: \n", len(cropped_pressure))
        avg_pressure = np.average(cropped_pressure)
        if len(cropped_pressure) ==0:
            break

        backwards_diff = []
        h = 2
        for j in range(2 * h, (len(filtered_force))):
            diff = ((3 * filtered_force[j]) - (4 * filtered_force[j - h]) + filtered_force[j - (2 * h)]) / (2 * h)
            backwards_diff.append(diff)
            j += 1

        cropped_backward_diff = np.average(np.array(backwards_diff))
        if filtered_force[0] >= FORCE_THRESHOLD:
            if float(
                    cropped_backward_diff) <= BACKWARD_DIFF_LIMIT and avg_pressure < PRESSURE_THRESHOLD:
                type = 1
                # print(f'Apple has been picked! Bdiff: {cropped_backward_diff}   Pressure: {avg_pressure}.\
                # Force: {filtered_force[0]} vs. Max Force: {np.max(force)}')
                break

            elif float(cropped_backward_diff) <= BACKWARD_DIFF_LIMIT and avg_pressure >= PRESSURE_THRESHOLD:
                type = 0
                # print(f'Apple was failed to be picked :( Force: {np.round(filtered_force[0])} Max Force: {np.max(force)}  Bdiff: {cropped_backward_diff}  Pressure: {avg_pressure}')
                break

            elif float(cropped_backward_diff) > BACKWARD_DIFF_LIMIT and np.round(filtered_force[0]) >= FORCE_THRESHOLD:
                # idx is incremented if the backward difference or pressure prevents immediate classification
                # (keeps track of intermediate conditions and controls loop flow based on whether thresholds are repeatedly unmet)
                idx = idx + 1
                i = i + 1

        else:
            if idx == 0: # if idx was NOT incremented previously, i is incremented and the loop continues.
                i = i + 1

            else: # If idx WAS incremented previously, the loop continues to see if picking type can be determined
                if float(cropped_backward_diff) > BACKWARD_DIFF_LIMIT and np.round(filtered_force[0]) < FORCE_THRESHOLD:
                    type = 0
                    # print(f'Apple was failed to be picked :( Force: {np.round(filtered_force[0])} Max Force: {np.max(force)}  Bdiff: {cropped_backward_diff}  Pressure: {avg_pressure}')
                    break

                elif avg_pressure >= PRESSURE_THRESHOLD and np.round(filtered_force[0]) < FORCE_THRESHOLD:
                    type = 0
                    # print(f'Apple was failed to be picked :( Force: {np.round(filtered_force[0])} Max Force: {np.max(force)}  Bdiff: {cropped_backward_diff}  Pressure: {avg_pressure}')
                    break

    return type, i


# Olivia's first plot
def plot_dx_over_time(time, dx, idx2, turn):
    plt.plot(time, dx)
    plt.plot(time[idx2 + INDEX_OG], dx[idx2 + INDEX_OG], "x")  # ax[2]
    plt.plot(time[turn[0]], dx[turn[0]], "x")
    plt.show()

# Will need to be updated based on how new data files are formatted
def extract_number_attempt_actual(pattern = r"2023111_realapple(\d+)_mode_dual_attempt_(\d+)_orientation_0_yaw_0"):
    # get the number and attempt of every bag filename matching the regex pattern
    results = []
    pattern_bag = pattern + r"\.bag"
    # Traverse the directory looking for matching files
    for filename in os.listdir(DIRECTORY):
        match = re.match(pattern_bag, filename)
        if match:
            # Extract number and attempt from the match groups
            number = int(match.group(1))
            attempt = int(match.group(2))

            # Check for corresponding JSON file
            json_filename = filename.replace(".bag", ".json")
            json_path = os.path.join(DIRECTORY, json_filename)

            # Initialize the actual value for apple pick result
            actual_value = None

            # If JSON file exists, open and parse it
            if os.path.isfile(json_path):
                with open(json_path, 'r') as json_file:
                    json_data = json.load(json_file)
                    # Extract the apple pick result from JSON
                    apple_pick_result = json_data.get("labels", {}).get("apple pick result", None)

                    # Set actual_value based on apple pick result
                    if apple_pick_result in ("a", "d", "e"):
                        actual_value = 1
                    elif apple_pick_result in ("b", "c"):
                        actual_value = 0

            # Append number, attempt, and actual value to the results list
            results.append((number, attempt, actual_value))
    print(f"datasets to analyze (number, attempt, actual): {results}\n")
    return results


def olivia_main():
    os.chdir(DIRECTORY)
    all_attempts = extract_number_attempt_actual()
    total_count_attempts = 0
    total_correct_classifications = 0
    positive_positive = 0 # positive means successful pick
    negative_negative = 0 # negative means failed pick
    positive_negative = 0  # position classification, negative actual
    negative_positive = 0  # negative classification, positive actual

    for attempt in all_attempts:
        number, attempt, actual = attempt
        attempt_name = f'2023111_realapple{number}_mode_dual_attempt_{attempt}_orientation_0_yaw_0'  # 2023111_realapple20_mode_dual_attempt_2_orientation_0_yaw_0
        print(f"attempt filename: {attempt_name}")
        try:
            classification = classification_from_file(attempt_name, number, attempt, actual)[0]
        except Exception as e:
            print(f"{RED}Error processing {attempt_name}: {e}{RESET}")
            continue
        # print(f"{RESET if correct else RED}{'Correct ' if correct else 'False '}classification")
        # print(f"classification: {classification}, actual: {actual}")
        if classification:
            if actual:
                positive_positive += 1
                total_correct_classifications += 1
            if not actual:
                positive_negative += 1
        if not classification:
            if actual:
                negative_positive += 1
            if not actual:
                negative_negative += 1
                total_correct_classifications += 1
        total_count_attempts += 1
    print(f"total correct: {total_correct_classifications} out of {total_count_attempts}\n{100*total_correct_classifications/total_count_attempts}% classification success rate")
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




# Olivia's classification program
def classification_from_file(name, number, attempt, actual):
    file = bag_to_csv(name)
    # separates the data from .bag into force and time arrays
    data = np.loadtxt('./' + file + '.csv', dtype="float", delimiter=',')

    sec = data[:, -1]
    nsec = data[:, -2]
    force = data[:, :-2]
    time = sec.tolist()
    ntime = nsec.tolist()

    # normalizes force data
    f_arr = np.linalg.norm(force, axis=1)

    # calculates elapsed time.
    tot_time = total_time(time, ntime)
    etime_force = elapsed_time(tot_time, tot_time)

    # extracts robot displacement data (total_disp) and joint times (joint_times_sec and joint_times_nsec)
    path = UR5e_ros1(0, name)
    total_disp = path.z
    joint_times_sec = path.times_seconds
    joint_times_nsec = path.times_nseconds

    tot_time_j = total_time(joint_times_sec, joint_times_nsec)
    etime_joint = elapsed_time(tot_time_j, tot_time_j)

    # calculates pressure data, normalizes it, and aligns it with the force data for analysis
    pressure_array = bag_pressure(file)
    p_arr = np.linalg.norm(pressure_array, axis=1)
    etimes_pressure = pressure_time(file)

    f_arr_col = f_arr[..., None]
    total_disp_col = np.array(total_disp)[..., None]
    p_arr_col = p_arr[..., None]

    # match_times synchronizes force, displacement, and pressure arrays by matching elapsed times.
    final_force, delta_x, general_time = match_times(etime_force, etime_joint, f_arr_col, total_disp_col)
    final_pressure, p_dis, other_time = match_times(etimes_pressure, etime_joint, p_arr_col, total_disp_col)
    fp = final_pressure.tolist()
    filtered = filter_force(final_force, 21)

    # A backward finite difference is applied to compute derivatives, giving a rate of change for filtered force data.
    backwards_diff = []
    h = 2
    for j in range(2 * h, (len(filtered))):
        diff = ((3 * filtered[j]) - (4 * filtered[j - h]) + filtered[j - (2 * h)]) / (2 * h)
        backwards_diff.append(diff)
        j += 1


    # Filtering is applied to the force array using a Butterworth low-pass filter to smooth out high-frequency noise
    low_bdiff = butter_lowpass_filter(backwards_diff, CUTOFF, FS, ORDER)

    # The filtered data (low_delta_x) is further smoothed using a Savitzky-Golay filter
    low_delta_x = scipy.signal.savgol_filter(delta_x[:, 0], 600, 1)

    # The script uses KneeLocator to identify concave and convex bends in the displacement data.
    # NOT SURE WHY SOME OF OLIVIA'S CODE IS COMMENTED OUT AND WHY SOME VARIABLES NEVER GET USED??
    if number != 25:
        kn = KneeLocator(general_time, low_delta_x, curve='concave', direction='decreasing')
        kn1 = KneeLocator(general_time, low_delta_x, curve='convex', direction='decreasing')
        idx2 = kn1.minima_indices[0]
        idx = kn1.minima_indices[-2]
        peaks, peak_index = scipy.signal.find_peaks(low_delta_x, height=0.03, plateau_size=(None, None))
        peak = int(peak_index['right_edges'][-1])
        peak1 = int(peak_index['left_edges'][0])

        # pressure turn
        pturn1 = np.where(final_pressure == np.min(final_pressure))[0]

        idx2 = np.where(low_delta_x == np.max(low_delta_x[idx2:-1]))[0][0]
        turn = np.where(low_delta_x == np.min(low_delta_x[idx2:-1]))[
            0]  # this is not picking where the turn around happens exactly
        turn2 = np.where(low_delta_x == np.max(low_delta_x[idx2:-1]))[0]

        cropped_x = delta_x[idx2 + INDEX_OG:turn[0]]
        new_x_part = flipping_data(cropped_x)

    else:
        idx2 = np.where(low_delta_x == np.min(low_delta_x))[0][0]
        turn = np.where(low_delta_x == np.max(low_delta_x[idx2:-1]))[0]
        new_x_part = delta_x[idx2 + INDEX_OG:turn[0]]

    # Uncomment below to visualize selected indices and turning points for analysis.
    # plot_dx_over_time(general_time, low_delta_x, idx2, turn)

    cropped_f = filtered[idx2 + INDEX_OG:turn[0]]
    # print("CROPPED_F length:", len(cropped_f))
    cropped_p = fp[idx2 + INDEX_OG: turn[0]]
    # print("CROPPED_P length:", len(cropped_p))
    cropped_low_bdiff = low_bdiff.tolist()[idx2 - 50 + INDEX_OG:turn[0] - 50]
    # print("CROPPED_LOW_BDIFF length:", len(cropped_low_bdiff))

    type, i = picking_type_classifier(cropped_f, cropped_p)

    # Uncomment following line to view the final plots of force, pressure v distance and the numerical derivative
    create_nforce_distance_num_deriv_plots(cropped_f, new_x_part, cropped_low_bdiff, cropped_p, i, number, attempt, type, general_time, actual)

    print("completed this file attempt:")
    print(f"{RED if type != actual else ''}type: {"Successful" if type else "Failed"}, i: {i}, actual: {"Successful\n" if actual else "Failed\n"}{RESET}")
    return type, actual