import matplotlib.pyplot as plt
import numpy as np

class StatsUtilities:

    @staticmethod
    def plot_graph(x: np.ndarray, y: np.ndarray, title: str = 'Performance Graph'):
        plt.plot(x, y)
        plt.xlabel("Episodes")
        plt.ylabel("Mean Reward")
        plt.title(title)
        plt.grid(True)
        plt.savefig(r"D:\Documentos\Pasta Projetos\Hollow-Knight-Neural-Network\assets\performance_graph.png", dpi=300)
        plt.show()