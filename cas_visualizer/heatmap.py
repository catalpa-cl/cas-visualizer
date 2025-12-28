import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Sample longer text (replace with any long string)
text = (
    "In computer science, artificial intelligence (AI), sometimes called machine intelligence, "
    "is intelligence demonstrated by machines, in contrast to the natural intelligence displayed "
    "by humans and animals. Leading AI textbooks define the field as the study of 'intelligent agents': "
    "any device that perceives its environment and takes actions that maximize its chance of successfully "
    "achieving its goals."
)

tokens = text.split()
feature_word = 'intelligence'  # Feature to search for

# Mark positions where feature_word appears (case-insensitive)
feature_positions = [1 if token.lower().strip(".,':;()") == feature_word else 0 for token in tokens]

# Set grid dimensions: e.g., 20 columns per row
cols = 20
rows = int(np.ceil(len(tokens) / cols))
heatmap_data = np.zeros((rows, cols))

for idx, val in enumerate(feature_positions):
    row = idx // cols
    col = idx % cols
    heatmap_data[row, col] = val

# Custom colormap: white for 0 (not found), blue for 1 (found)
cmap = ListedColormap(['white', '#0077FF'])

plt.figure(figsize=(cols * 0.3, rows * 0.3))  # Small fixed-size cells
plt.imshow(heatmap_data, cmap=cmap, interpolation='none', aspect='equal')
plt.colorbar(ticks=[0,1], label=f"Occurrence of '{feature_word}'")
plt.title(f"Heatmap: Positions of '{feature_word}'")
plt.xlabel("Token Position (column)")
plt.ylabel("Token Row")
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.show()