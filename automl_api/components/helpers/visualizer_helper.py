import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tempfile

class VisualizerHelper:
    def __init__(self):
        self.temp_dir = tempfile.gettempdir()

    def _save_plot(self, filename):
        file_path = os.path.join(self.temp_dir, filename)
        plt.savefig(file_path)
        plt.close()
        return file_path

    def plot_histogram(self, data: pd.DataFrame, column: str, bins: int = 20, title: str = None) -> str:
        plt.figure(figsize=(8, 6))
        sns.histplot(data=data[column], bins=bins, kde=True)
        plt.title(title if title else f"Histogram of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")

        return self._save_plot(f"{column}_histogram.png")

    def plot_scatter(self, data: pd.DataFrame, x: str, y: str, title: str = None) -> str:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=data[x], y=data[y])
        plt.title(title if title else f"Scatter plot of {x} vs {y}")
        plt.xlabel(x)
        plt.ylabel(y)

        return self._save_plot(f"{x}_vs_{y}_scatter.png")

    def plot_correlation_matrix(self, data: pd.DataFrame, cmap: str = 'coolwarm') -> str:
        plt.figure(figsize=(10, 8))
        corr = data.corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cbar=True, cmap=cmap)
        plt.title(f"Correlation Matrix")

        return self._save_plot(f"Correlation Matrix.png")