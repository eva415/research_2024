import os
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from rf_pick_classification_funcs import filter_force
from feature_importance_augment_rf_pick_classifier import (
    return_flex_array, return_pressure_array, return_force_array, return_tof_array
)

# ----- Labels -----
STATE_PICKING  = 0
STATE_SUCCESS  = 1
STATE_PRE_FAIL = 3
STATE_FAIL     = 2

# ----- MAIN -----
if __name__ == "__main__":
    bag_dir = "/home/imml/Desktop/single_pick_trial"
    # bag_name = "final_approach_and_pick_20251029_180739"
    bag_name = "final_approach_and_pick_20251030_135929"

    # ---- Load signals ----
    raw_f_arr, f_arr, etime_force, labels = return_force_array(
        f"{bag_dir}/{bag_name}.db3force.csv"
    )
    p_arr, _ = return_pressure_array(f"{bag_dir}/{bag_name}.db3pressure.csv")
    tof_arr, _ = return_tof_array(f"{bag_dir}/{bag_name}.db3tof.csv")
    flex_arr, _ = return_flex_array(f"{bag_dir}/{bag_name}.db3_flex_norm.csv")

    # ---- Resample to force length ----
    flex_arr = scipy.signal.resample(flex_arr, len(f_arr))
    p_arr    = scipy.signal.resample(p_arr, len(f_arr))
    tof_arr  = scipy.signal.resample(tof_arr, len(f_arr))
    f_arr    = filter_force([f_arr], 21)[0]

    # ---- Find transition indices ----
    transitions = []
    for i in range(1, len(labels)):
        if labels[i-1] == STATE_PICKING and labels[i] == STATE_SUCCESS:
            transitions.append((i, "Pick Success"))
        elif labels[i-1] == STATE_PRE_FAIL and labels[i] == STATE_FAIL:
            transitions.append((i, "Slip / Fail"))

    print(f"Found {len(transitions)} transitions")

    # ---- Plot ±1 second around each transition ----
    window_sec = 1.0

    pdf_name = f"{bag_dir}/{bag_name}_sensor_windows.pdf"
    with PdfPages(pdf_name) as pdf:
        for idx, event_name in transitions:
            t0 = etime_force[idx]

            mask = (etime_force >= t0 - window_sec) & \
                   (etime_force <= t0 + window_sec)

            t_rel = etime_force[mask] - t0

            fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

            axs[0].plot(t_rel, f_arr[mask], lw=3.5)
            axs[1].plot(t_rel, p_arr[mask], lw=3.5)
            axs[2].plot(t_rel, flex_arr[mask], lw=3.5)
            axs[3].plot(t_rel, tof_arr[mask], lw=3.5)


            axs[0].set_ylabel("Force", fontsize=30)
            axs[1].set_ylabel("Pressure", fontsize=30)
            axs[2].set_ylabel("Flex", fontsize=30)
            axs[3].set_ylabel("TOF", fontsize=30)
            # axs[3].set_xlabel("Time relative to event (s)", fontsize=22)

            for ax in axs:
                ax.axvline(0, color="k", linestyle="--", alpha=0.6, lw=3.5)
                ax.tick_params(axis='both', which='both', labelbottom=False, labelleft=False)


            # fig.suptitle(
            #     f"{event_name} — ±1s Sensor Signals",
            #     fontsize=26
            # )

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Saved sensor window plots to {pdf_name}")
