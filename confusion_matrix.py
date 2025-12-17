import matplotlib.pyplot as plt, numpy as np
from PIL import Image
import os

def merge_pngs_to_pdf(directory, file_names, output_pdf):
    images = []
    
    for file_name in file_names:
        file_path = os.path.join(directory, file_name)
        img = Image.open(file_path)
        # Convert to RGB if it's RGBA (needed for PDF)
        if img.mode == "RGBA":
            img = img.convert("RGB")
        images.append(img)
    
    if images:
        # Save all images into a single PDF
        images[0].save(output_pdf, save_all=True, append_images=images[1:])
        print(f"PDF created successfully: {output_pdf}")
    else:
        print("No images to merge!")

directory = "/home/imml/Desktop"
file_names = ["confusion_FTF.png", "confusion_FPT.png", "confusion_FPF.png", "confusion_FPTF.png"]
output_pdf = os.path.join(directory, "merged_confusion2.pdf")

merge_pngs_to_pdf(directory, file_names, output_pdf)


def make_confusion_matrix(name, pos_pos, neg_neg, pos_neg, neg_pos):
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
        ax.text(j, i, f'{val}', ha='center', va='center', color=cell_color, fontsize=20)

    # Add gridlines for clearer separation
    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=2)

    plt.title(f"Confusion Matrix: {name}", fontsize=22)
    plt.show()


# pos_pos → True Positive (TP)
# pos_neg → False Negative (FN)
# neg_pos → False Positive (FP)
# neg_neg → True Negative (TN)
positive_positive = 1 + 59
positive_negative = 8 + 4
negative_positive = 1
negative_negative = 10
# make_confusion_matrix("Force, Pressure, ToF, Flex", positive_positive, negative_negative, positive_negative, negative_positive)
