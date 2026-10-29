import matplotlib.pyplot as plt
import numpy as np
import os

from ..config import PROJECT_ROOT

class StatsUtilities:

    @staticmethod
    def plot_graph(x: np.ndarray, y: np.ndarray, title: str = 'Training Behavior'):
        plt.plot(x, y)
        plt.xlabel("Episodes")
        plt.ylabel("Mean Reward")
        plt.title(title)
        plt.grid(True)
        plt.savefig(os.path.join(PROJECT_ROOT, 'assets', 'performance_graph.png'), dpi=300)
        plt.show()