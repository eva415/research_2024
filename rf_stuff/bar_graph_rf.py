import matplotlib.pyplot as plt
import numpy as np

# Model names
models = ["F", "F, T", "F, P", "F, f", "F, P, T", "F, T, f", "F, P, f", "F, P, f, T"]

# Overall accuracy (from my logs)
accuracy = [.6977, .7982, .8204, .8489, .8489, .8822, .8894, .9136]

# Macro-average precision and recall (from classification reports, can also use weighted avg)
# precision = [0.7398, 0.8486, 0.8746, 0.8634, 0.8971, 0.8969, 0.9046, 0.9251]
# recall = [0.6743, 0.7811, 0.8011, 0.8427, 0.8423, 0.8811, 0.8887, 0.9184]

# WEIGHTED AVGS  
precision = [0.7153, 0.8179, 0.8510, 0.8500, 0.8690, 0.8833, 0.8903, 0.9135]
recall = [0.6977, 0.7982, 0.8204, 0.8489, 0.8489, 0.8822, 0.8894, 0.9136]

# Bar width and positions
x = np.arange(len(models))
width = 0.25

fig, ax = plt.subplots(figsize=(12,6))

# Plot bars
ax.bar(x - width, precision, width, label='Precision', color='skyblue')
ax.bar(x, recall, width, label='Recall', color='lightgreen')
ax.bar(x + width, accuracy, width, label='Accuracy', color='salmon')

# Labels, title, legend
ax.set_ylabel('Score')
ax.set_title(f'Random Forest Model Performance Comparison \n(F = Force, T = ToF, P = Pressure, f = flex)')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45, ha='right')
ax.set_ylim(0.6, 1.0)
ax.legend(loc = "upper left")


# Optionally, add value labels on top
for i in range(len(models)):
    ax.text(x[i] - width, precision[i]+0.02, f"{precision[i]:.2f}", ha='center', fontsize=8)
    ax.text(x[i], recall[i]+0.02, f"{recall[i]:.2f}", ha='center', fontsize=8)
    ax.text(x[i] + width, accuracy[i]+0.02, f"{accuracy[i]:.1f}", ha='center', fontsize=8)

plt.tight_layout()
plt.show()
