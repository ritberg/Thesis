import umap
import numpy as np
from loading import loading
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.colors as mcolors

# Define the t-SNE algorithm using 
# the package scikit-learn. We set `n_components` to 2 
# because we want to output a bidimensional plot.
reducer = TSNE(n_components=2)

# Define the authors' list.
a = ["Afanassiev",
    "Federovsky",
    "Gran",
    "Jurgenson",
    "Makine",
    "Modiano",
    "Quignard",
    "Duras",
    "Ernaux",
    "Houellebecq"
]
# Define the dictionary to map the author 
# name to its book index.
books = {
    "Afanassiev": [1],
    "Federovsky": [2],
    "Gran": [3],
    "Jurgenson": [4],
    "Makine": [9],
    "Modiano": [18],
    "Quignard": [19],
    "Duras": [20],
    "Ernaux": [21, 22],
    "Houellebecq": [23],
}

# Load the features and the targets 
# with the functions loading described before.

features, targets, _, _, keys = loading(books,
										  "dataMixed_1",
										  start_range=300,
										  end_range=1300,
										  true_disputed_class=1,
										  segment_length=400
										  )
# Standardize the features with the Standard Scaler.
scaled_features = StandardScaler().fit_transform(features)

# Apply t-SNE to reduce the number dimensions to 2 dimensions.
tsne_projection = reducer.fit_transform(features.astype(np.float64))

# Define the colors to be used to plot the tsne output.
colors = [mcolors.BASE_COLORS["r"], mcolors.BASE_COLORS["g"],
          mcolors.BASE_COLORS["b"],
          mcolors.BASE_COLORS["c"], mcolors.BASE_COLORS["m"],
          mcolors.BASE_COLORS["y"],
          mcolors.BASE_COLORS["k"], mcolors.CSS4_COLORS["blueviolet"],
          mcolors.CSS4_COLORS["navy"], mcolors.CSS4_COLORS["darkolivegreen"]]

# Plot the result
plt.figure()
for i in range(10):
    plt.scatter(
        tsne_projection[np.where(targets==i), 0],
        tsne_projection[np.where(targets==i), 1],
        c=colors[int(i)], label=a[i])
plt.legend(fontsize=10, frameon=1, #loc='best',
           bbox_to_anchor=(1.1, -0.1),
           fancybox=True, shadow=False, ncol=4)
plt.savefig("tSNE.png", bbox_inches='tight', dpi=800)
