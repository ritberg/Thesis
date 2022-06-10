import numpy as np
import matplotlib.pyplot as plt
from typing import List


def ridge_bar_plot(weights: np.ndarray,
                   keys: List[str],
                   title: str):
    plt.figure(figsize=(20, 5))
    plt.xticks(rotation=90)
    plt.subplots_adjust(bottom=0.4)
    plt.bar(keys, weights)
    plt.savefig(title)

def horizontal_bar_plot(weights: np.ndarray,
                   keys: List[str],
                   title: str):
    plt.figure(figsize=(5,20))
    plt.xticks(rotation=90, fontsize=14)
    plt.yticks(fontsize=14)
    for weight, key in zip(weights, keys):
        color = "lightblue" if weight > 0 else "lightsalmon"
        plt.barh(key, weight, color=color, height=0.7)

    plt.tight_layout()
    plt.savefig(title)
