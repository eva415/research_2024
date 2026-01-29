# MLP based on my RF implementation:
# keeping augmentation identical
# keeping windowing identical
# keeping bag-level splits identical (fixed split tagging)
# keeping plotting logic unchanged
# keeping permutation importance logic intact
# ADDING mandatory scaling

import os
import numpy as np
import scipy.signal
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from rf_pick_classification_funcs import total_time, elapsed_time, filter_force
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import medfilt
import random
import pandas as pd
from sklearn.inspection import permutation_importance
import seaborn as sns
from feature_importance_augment_rf_pick_classifier import (
    augment_paper,
    process_csv_files,
    load_bags,
    create_windowed_samples,
    generate_windowed_data_from_bags,
    return_tof_array,
    return_force_array,
    return_pressure_array,
    return_flex_array,
    plot_bag_file
)
import matplotlib as mpl

mpl.rcParams.update({
    "font.size": 20,            # base font size
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "figure.titlesize": 24,
    "lines.linewidth": 3,})

STEP_SIZE = 5
WINDOW_SIZE = 5

# ----- Label definitions -----
STATE_PICKING  = 0
STATE_SUCCESS  = 1
STATE_PRE_FAIL = 3
STATE_FAIL     = 2

# ---------- MAIN ----------
if __name__ == "__main__":

    failed_dir = "/home/imml/Desktop/failed_picks"
    success_dir = "/home/imml/Desktop/successful_picks"

    # ----- Load bags with augmentation -----
    failed_bags = sorted(
        load_bags(failed_dir, augment_failed=True, num_aug_failed=7),
        key=lambda x: x[0]
    )
    success_bags = sorted(
        load_bags(success_dir, augment_success=True, num_aug_success=1),
        key=lambda x: x[0]
    )

    all_bags = failed_bags + success_bags
    random.seed(45)
    random.shuffle(all_bags)

    # ----- Train/val/test split -----
    n = len(all_bags)
    train_bags = all_bags[:int(0.8 * n)]
    val_bags   = all_bags[int(0.8 * n):int(0.9 * n)]
    test_bags  = all_bags[int(0.9 * n):]

    # ----- Add split tags -----
    train_bags = [(p,f,l,"TRAIN") for (p,f,l) in train_bags]
    val_bags   = [(p,f,l,"VAL")   for (p,f,l) in val_bags]
    test_bags  = [(p,f,l,"TEST")  for (p,f,l) in test_bags]

    all_bags = train_bags + val_bags + test_bags

    # ----- Generate windowed datasets -----
    X_train, y_train = generate_windowed_data_from_bags(train_bags)
    X_val,   y_val   = generate_windowed_data_from_bags(val_bags)
    X_test,  y_test  = generate_windowed_data_from_bags(test_bags)

    # ----- Mandatory scaling -----
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # ----- Train MLP -----
    clf = MLPClassifier(
        hidden_layer_sizes=(150, 50),
        activation="relu",
        alpha=1e-4,
        batch_size=250,
        max_iter=200,
        early_stopping=True,
        random_state=42,
        verbose=True
    )
    clf.fit(X_train, y_train)

    # ----- Global permutation importance -----
    sensor_names = ["Flex", "Pressure", "Force", "TOF"]
    n_sensors = len(sensor_names)
    n_timesteps = WINDOW_SIZE

    feature_labels = [f"{s}_t{t}" for s in sensor_names for t in range(n_timesteps)]

    r = permutation_importance(
        clf, X_test, y_test,
        n_repeats=30,
        random_state=42,
        n_jobs=-1
    )
    perm = r.importances_mean

    feat_df = pd.DataFrame({"feature": feature_labels, "importance": perm})
    sensor_importance = (
        feat_df
        .assign(sensor=feat_df.feature.str.split("_").str[0])
        .groupby("sensor")["importance"]
        .sum()
        .sort_values(ascending=False)
    )

    print("\n=== Overall Sensor Importance ===")
    print(sensor_importance)

    # ----- Per-state permutation importance (MLP) -----
    states_to_check = [STATE_PICKING, STATE_PRE_FAIL, STATE_SUCCESS, STATE_FAIL]
    state_names = ["Picking", "Pre-Fail", "Success", "Fail"]

    # Store results in a DataFrame
    state_sensor_importance = pd.DataFrame(index=sensor_names, columns=state_names, dtype=float)

    for state, name in zip(states_to_check, state_names):
        idx = np.where(y_test == state)[0]
        if len(idx) == 0:
            continue

        X_state = X_test[idx]
        y_state = y_test[idx]

        # Permutation importance for this state
        r = permutation_importance(
            clf, X_state, y_state,
            n_repeats=30,        # increase repeats for MLP stability
            random_state=42,
            n_jobs=-1
        )
        perm_importances = r.importances_mean

        # Sum over timesteps for each sensor
        sensor_vals = []
        for i in range(n_sensors):
            start_idx = i * n_timesteps
            end_idx = start_idx + n_timesteps
            sensor_vals.append(np.sum(perm_importances[start_idx:end_idx]))

        state_sensor_importance[name] = sensor_vals

    # ----- Global normalization to ±1 -----
    max_abs_all = np.max(np.abs(state_sensor_importance.values))
    if max_abs_all > 0:
        state_sensor_importance /= max_abs_all

    print("\n=== Feature Importance per State (Normalized Globally to ±1) ===")
    print(state_sensor_importance)

    # ----- Heatmap -----
    plt.figure(figsize=(6,4))
    sns.heatmap(
        state_sensor_importance,
        annot=True,
        cmap="coolwarm",
        center=0,
        annot_kws={"size": 12}
    )

    plt.title(
        f"Sensor Feature Importance by State\n(MLP, Permutation Importance)",
        fontsize=22
    )
    plt.xlabel("State", fontsize=14)
    plt.ylabel("Sensor", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.tight_layout()
    # plt.show()

    # ----- Evaluation -----
    y_pred = clf.predict(X_test)

    print("\n===== MODEL PERFORMANCE (TEST SET) =====")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    # ----- Generate PDF plots (MLP predictions) -----
    pdf_name = "MLP_jan12.pdf"
    with PdfPages(pdf_name) as pdf:
        for prefix, _, _, split in test_bags:
            plot_bag_file(
                prefix,
                split=split,
                clf=clf,
                window_size=WINDOW_SIZE,
                step=STEP_SIZE,
                save_pdf=pdf,
                scaler=scaler
            )

    print(f"All bag plots saved to {pdf_name}")
