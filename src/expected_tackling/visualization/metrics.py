import os
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plotly.figure_factory import create_table

figures_path = str(Path(__file__).parents[3] / "reports/figures")


def plot_confusion_matrix(train_confusion_matrix, test_confusion_matrix, name="confusion_matrix"):
    train_table = create_table(
        pd.DataFrame(
            train_confusion_matrix,
            columns=["Predicted Negatives", "Predicted Positives"],
            index=["Actual Negatives", "Actual Positives"],
        ),
        index=True,
    )
    train_table.update_layout(
        autosize=False,
        width=500,
        height=200,
    )
    train_table.write_image(figures_path + f"/train_table.png", format="png")

    test_table = create_table(
        pd.DataFrame(
            test_confusion_matrix,
            columns=["Predicted Negatives", "Predicted Positives"],
            index=["Actual Negatives", "Actual Positives"],
        ),
        index=True,
    )
    test_table.update_layout(
        autosize=False,
        width=500,
        height=200,
    )
    test_table.write_image(figures_path + f"/test_table.png", format="png")

    train_image = Image.open(figures_path + f"/train_table.png")
    test_image = Image.open(figures_path + f"/test_table.png")

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    axes[0].imshow(np.array(train_image))
    axes[0].set_axis_off()
    axes[0].set_title("Train Confusion Matrix")
    axes[1].imshow(np.array(test_image))
    axes[1].set_axis_off()
    axes[1].set_title("Test Confusion Matrix")

    os.remove(figures_path + f"/train_table.png")
    os.remove(figures_path + f"/test_table.png")

    plt.savefig(figures_path + f"/{name}.png", bbox_inches="tight")
