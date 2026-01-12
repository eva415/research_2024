import os
import numpy as np
import scipy.signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from rf_pick_classification_funcs import total_time, elapsed_time, filter_force
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import medfilt
import random
from matplotlib.colors import to_rgb

STEP_SIZE = 10
WINDOW_SIZE = 50
print(f"step size: {STEP_SIZE}, window size: {WINDOW_SIZE}")

# ---------- Data loading functions ----------
def return_tof_array(csv_file):
    data = np.loadtxt(csv_file, delimiter=',', skiprows=1)
    raw_sec_array = data[:, -3]
    raw_nsec_array = data[:, -2]
    labels = data[:, -1]
    total_tof_time = total_time(raw_sec_array, raw_nsec_array)
    elapsed_tof_time = elapsed_time(total_tof_time)
    tof_values = data[:, :-3]
    return tof_values, elapsed_tof_time

def return_force_array(csv_file):
    data = np.loadtxt(csv_file, delimiter=',', skiprows=1)
    raw_sec_array = data[:, -3]
    raw_nsec_array = data[:, -2]
    labels = data[:, 5]
    total_force_time = total_time(raw_sec_array, raw_nsec_array)
    elapsed_force_time = elapsed_time(total_force_time)
    force_array = data[:, 0:3]
    fz_force_array = -force_array[:, 2]
    return force_array, fz_force_array, elapsed_force_time, labels

def return_pressure_array(csv_file):
    data = np.loadtxt(csv_file, delimiter=',', skiprows=1)
    raw_sec_array = data[:, -3]
    raw_nsec_array = data[:, -2]
    labels = data[:, -1]
    total_pressure_time = total_time(raw_sec_array, raw_nsec_array)
    elapsed_pressure_time = elapsed_time(total_pressure_time)
    pressure_array = data[:, :-3]
    return pressure_array, elapsed_pressure_time

def return_flex_array(csv_file, smoothing_method='median', param=5):
    data = np.loadtxt(csv_file, delimiter=',', skiprows=1)
    flex_norm = data[:, -1]
    raw_sec_array = data[:, -3]
    raw_nsec_array = data[:, -2]
    total_flex_time = total_time(raw_sec_array, raw_nsec_array)
    elapsed_flex_time = elapsed_time(total_flex_time)
    flex_smooth = medfilt(flex_norm, kernel_size=param) if smoothing_method=='median' else flex_norm
    return flex_smooth, elapsed_flex_time

def process_csv_files(filename):
    force_csv = f"{filename}.db3force.csv"
    pressure_csv = f"{filename}.db3pressure.csv"
    tof_csv = f"{filename}.db3tof.csv"
    flex_csv = f"{filename}.db3_flex_norm.csv"

    raw_f_arr, f_arr, etime_force, labels = return_force_array(force_csv)
    p_arr, _ = return_pressure_array(pressure_csv)
    tof_arr, _ = return_tof_array(tof_csv)
    flex_arr, _ = return_flex_array(flex_csv)

    flex_arr_new = scipy.signal.resample(flex_arr, len(f_arr))
    p_arr_new = scipy.signal.resample(p_arr, len(f_arr))
    filtered_force = filter_force([f_arr], 21)[0]
    tof_arr_new = scipy.signal.resample(tof_arr, len(f_arr))

    # ---- CROP OUT PRE-GRASP ----
    # find first index where pressure <= -56 (grasp starts)
    grasp_indices = np.where(p_arr_new <= -56)[0]

    if len(grasp_indices) > 0:
        first_grasp = grasp_indices[0]

        # crop all signals + labels after grasp begins
        flex_arr_new = flex_arr_new[first_grasp:]
        p_arr_new = p_arr_new[first_grasp:]
        filtered_force = filtered_force[first_grasp:]
        tof_arr_new = tof_arr_new[first_grasp:]
        labels = labels[first_grasp:]


    # features = np.column_stack((flex_arr_new, p_arr_new, filtered_force, tof_arr_new)) # all sensors # REPLACE HERE
    features = filtered_force.reshape(-1, 1) # just force
    # features = np.column_stack((p_arr_new, filtered_force)) # force and pressure
    # features = np.column_stack((tof_arr_new, filtered_force)) # force and ToF
    # features = np.column_stack((flex_arr_new, filtered_force)) # force and flex
    # features = np.column_stack((tof_arr_new, p_arr_new, filtered_force)) # f, p, t
    # features = np.column_stack((flex_arr_new, p_arr_new, filtered_force)) # f, p, f
    # features = np.column_stack((flex_arr_new, tof_arr_new, filtered_force)) # f, t, f

    return features, labels

def create_windowed_samples(X, y, window_size=WINDOW_SIZE, step=1):
    n_timesteps, n_features = X.shape
    X_windows, y_windows = [], []
    for start in range(0, n_timesteps - window_size + 1, step):
        end = start + window_size
        X_windows.append(X[start:end].flatten())
        y_windows.append(y[end - 1])
    return np.array(X_windows), np.array(y_windows)

# ---------- Bag-level loading ----------
def load_bags(directory):
    bags = []
    for file in os.listdir(directory):
        if file.endswith(".csv") and ".db3force" in file:
            prefix = os.path.join(directory, file.replace(".db3force.csv", ""))
            features, labels = process_csv_files(prefix)
            bags.append((prefix, features, labels))
    return bags

def generate_windowed_data_from_bags(bags, window_size=WINDOW_SIZE, step=STEP_SIZE):
    X_list, y_list = [], []
    # modified to handle split-tagged bags (ignore the 4th element)
    for _, features, labels, _ in bags:   ### ADDED
        X_w, y_w = create_windowed_samples(features, labels, window_size, step)
        X_list.append(X_w)
        y_list.append(y_w)
    X_all = np.vstack(X_list)
    y_all = np.hstack(y_list)
    return X_all, y_all


            # filtered_force.reshape(-1, 1),   # force-only
            # np.column_stack((p_arr_new, filtered_force)), # force and pressure
            # np.column_stack((tof_arr_new, filtered_force)), # force and tof

# ---------- Plotting ----------
def plot_bag_file(filename, split="UNKNOWN", clf=None, window_size=WINDOW_SIZE, step=STEP_SIZE, save_pdf=None):
    # ---- Load signals ----
    raw_f_arr, f_arr, etime_force, labels = return_force_array(f"{filename}.db3force.csv")
    p_arr, _ = return_pressure_array(f"{filename}.db3pressure.csv")
    tof_arr, _ = return_tof_array(f"{filename}.db3tof.csv")
    flex_arr, _ = return_flex_array(f"{filename}.db3_flex_norm.csv")

    flex_arr_new = scipy.signal.resample(flex_arr, len(f_arr))
    p_arr_new = scipy.signal.resample(p_arr, len(f_arr))
    filtered_force = filter_force([f_arr], 21)[0]
    tof_arr_new = scipy.signal.resample(tof_arr, len(f_arr))

    # ---- CROP OUT PRE-GRASP ----
    grasp_indices = np.where(p_arr_new <= -56)[0]

    if len(grasp_indices) > 0:
        first_grasp = grasp_indices[0]

        flex_arr_new = flex_arr_new[first_grasp:]
        p_arr_new = p_arr_new[first_grasp:]
        filtered_force = filtered_force[first_grasp:]
        tof_arr_new = tof_arr_new[first_grasp:]
        labels = labels[first_grasp:]
        etime_force = etime_force[first_grasp:]

    # ---- RF prediction over full signal ----
    first_pick_idx, first_pick_val = None, None
    pred_full = None
    window_ranges = []      # <<< NEW (stores window correctness spans)

    if clf:
        X_windowed, _ = create_windowed_samples(
            # np.column_stack((flex_arr_new, tof_arr_new, filtered_force)), # f t f
            # np.column_stack((flex_arr_new, p_arr_new, filtered_force, tof_arr_new)),    # all sensors replace here
            # np.column_stack((tof_arr_new, p_arr_new, filtered_force)),    # f p t
            # np.column_stack((flex_arr_new, p_arr_new, filtered_force)), #fpf
            # np.column_stack((tof_arr_new, filtered_force)),  # f t
            # np.column_stack((p_arr_new, filtered_force)), # f p
            # np.column_stack((flex_arr_new, filtered_force)), # f f
            filtered_force.reshape(-1, 1), # just force
            labels, window_size, step
        )
        y_pred = clf.predict(X_windowed)

        pred_full = np.zeros_like(labels, dtype=int)
        for w, start in enumerate(range(0, len(labels)-window_size+1, step)):
            end = start + window_size
            pred_full[start:end] = y_pred[w]

        # ---- NEW: compute per-window correctness ----
        for w, start in enumerate(range(0, len(labels) - window_size + 1, step)):
            end = start + window_size
            true_label = labels[end - 1]
            pred_label = y_pred[w]
            correct = (true_label == pred_label)
            window_ranges.append((start, end, correct))

        # ---- Find first predicted pick ----
        for i in range(1, len(pred_full)):
            if pred_full[i-1] == 0 and pred_full[i] in [1,2]:
                first_pick_idx = i
                first_pick_val = pred_full[i]
                break

    # ---- Ground truth pick time ----
    true_pick_idx, true_pick_val = None, None
    for i in range(1, len(labels)):
        if labels[i-1] == 0 and labels[i] in [1,2]:
            true_pick_idx = i
            true_pick_val = labels[i]
            break

    # ---- Plot signals ----
    fig, axs = plt.subplots(4, 1, figsize=(12,8), sharex=True)

    # ---- NEW: window shading (green = correct, red = wrong) ----
    if clf:
        for (start, end, correct) in window_ranges:
            x0 = etime_force[start]
            x1 = etime_force[end-1] if end-1 < len(etime_force) else etime_force[-1]
            color = 'green' if correct else 'red'
            for ax in axs:
                ax.axvspan(x0, x1, color=color, alpha=0.15)

    axs[0].plot(etime_force, filtered_force, label="Force")
    axs[1].plot(etime_force, p_arr_new, label="Pressure")
    axs[2].plot(etime_force, flex_arr_new, label="Flex")
    axs[3].plot(etime_force, tof_arr_new, label="TOF")

    # ---- Highlight regions / lines ----
    legend_handles = []
    if true_pick_idx is not None:
        true_pick_time = etime_force[true_pick_idx]
        for ax in axs:
            line = ax.axvline(true_pick_time, color='black', linestyle='--', linewidth=2)
        legend_handles.append(line)

    if first_pick_idx is not None:
        color = 'green' if first_pick_val == 1 else 'red'
        predicted_pick_time = etime_force[first_pick_idx]
        for ax in axs:
            line = ax.axvline(predicted_pick_time, color=color, linestyle=':', linewidth=2)
        legend_handles.append(line)

    # ---- Axis labels ----
    axs[0].set_ylabel("Force")
    axs[1].set_ylabel("Pressure")
    axs[2].set_ylabel("Flex")
    axs[3].set_ylabel("TOF")
    axs[3].set_xlabel("Time (s)")

    # ---- Title ----
    suptitle_str = f"{os.path.basename(filename)}   [{split}]\nTrue label: {'Success' if int(labels[-1])==1 else 'Fail'}"
    if true_pick_idx is not None:
        suptitle_str += f" | True pick time: {etime_force[true_pick_idx]:.2f}s"
    if first_pick_idx is not None:
        suptitle_str += f"\nRF label: {'Success' if int(first_pick_val)==1 else 'Fail'} | RF pick time: {predicted_pick_time:.2f}s"
    else:
        suptitle_str += "\nRF label: Unclassified | RF pick time: NONE"
    fig.suptitle(suptitle_str, fontsize=14)

    if legend_handles:
        fig.legend(legend_handles,
                   ["True pick/slip time", "Predicted pick/slip time"],
                   loc='upper right')

    plt.tight_layout(rect=[0,0,1,0.96])
    # plt.show()
    if save_pdf:
        save_pdf.savefig(fig)
    plt.close(fig)

# ---------- Main ----------
if __name__ == "__main__":
    failed_dir = "/home/imml/Desktop/failed_picks"
    success_dir = "/home/imml/Desktop/successful_picks"
    window_size = WINDOW_SIZE
    step = STEP_SIZE

    # ---------- Deterministic bag loading + shuffle ----------
    failed_bags  = sorted(load_bags(failed_dir),   key=lambda x: x[0])
    success_bags = sorted(load_bags(success_dir), key=lambda x: x[0])

    all_bags = failed_bags + success_bags

    # Fixed seed ensures deterministic split every run
    random.seed(45)
    random.shuffle(all_bags)

    # Bag-level split
    n_total = len(all_bags)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    train_bags = all_bags[:n_train]
    val_bags = all_bags[n_train:n_train+n_val]
    test_bags = all_bags[n_train+n_val:]

    ### TAG BAGS WITH SPLIT NAME
    train_bags = [(p,f,l,"TRAIN") for (p,f,l) in train_bags]
    val_bags   = [(p,f,l,"VAL")   for (p,f,l) in val_bags]
    test_bags  = [(p,f,l,"TEST")  for (p,f,l) in test_bags]

    all_bags = train_bags + val_bags + test_bags

    # Windowed datasets
    X_train, y_train = generate_windowed_data_from_bags(train_bags, window_size, step)
    X_val, y_val = generate_windowed_data_from_bags(val_bags, window_size, step)
    X_test, y_test = generate_windowed_data_from_bags(test_bags, window_size, step)

    total_samples = len(X_train)+len(X_val)+len(X_test)
    print(f"Total samples: {total_samples}")
    print(f"Training samples: {len(X_train)} ({len(X_train)/total_samples*100:.1f}%)")
    print(f"Validation samples: {len(X_val)} ({len(X_val)/total_samples*100:.1f}%)")
    print(f"Test samples: {len(X_test)} ({len(X_test)/total_samples*100:.1f}%)")

    # Train Random Forest
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # ---------- EVALUATION ----------
    y_pred = clf.predict(X_test)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"\n===== MODEL PERFORMANCE (TEST SET) =====")
    print(f"Accuracy: {100*acc:.2f}%\n")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix (Counts):")
    print(cm)

    # Classification Report (precision, recall, F1)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    # --- Optional Confusion Matrix Plot ---
    # fig, ax = plt.subplots(figsize=(5,4))
    # im = ax.imshow(cm, interpolation='nearest')
    # ax.figure.colorbar(im, ax=ax)

    # ax.set_title("Confusion Matrix (Test Set)")
    # ax.set_xlabel("Predicted Label")
    # ax.set_ylabel("True Label")

    # ax.set_xticks(np.arange(cm.shape[1]))
    # ax.set_yticks(np.arange(cm.shape[0]))
    # ax.set_xticklabels([0,1,2])
    # ax.set_yticklabels([0,1,2])

    # plt.tight_layout()
    # plt.show()

    # Generate PDF plots
    with PdfPages("RF_f_25.pdf") as pdf: # REPLACE HERE
        for prefix, _, _, split in all_bags:
            plot_bag_file(prefix, split=split, clf=clf,
                          window_size=window_size, step=step,
                          save_pdf=pdf)

    print("All bag plots saved")
