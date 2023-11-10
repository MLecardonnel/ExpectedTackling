import os
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from shapash import SmartExplainer

figures_path = str(Path(__file__).parents[3] / "reports/figures")


class Explainer:
    def __init__(self, features_data, model, sample_size=100000):
        y = features_data["will_tackle"].astype(int)
        X = features_data.drop(
            columns=["gameId", "playId", "nflId", "frameId", "x", "y", "playDirection", "will_tackle"]
        )

        X_sample = X.sample(sample_size)

        xpl = SmartExplainer(
            model=model,
        )

        xpl.compile(
            x=X_sample,
            y_target=y.loc[X_sample.index],
        )

        self.xpl = xpl

    def run_app(self, port=8050):
        return self.xpl.run_app(port=port)

    def plot_contributions_examples(
        self,
        features_columns=[
            "distance_to_ball_carrier",
            "direction_to_ball_carrier",
            "s",
            "ball_carrier_distance_to_endzone",
        ],
        name="contributions_examples",
    ):
        nb_features = len(features_columns)
        nrows = nb_features // 2 + nb_features % 2
        fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(12, 4 * nrows))
        for i in range(nb_features):
            col = features_columns[i]
            self.xpl.plot.contribution_plot(col=col).write_image(figures_path + f"/{col}.png", format="png")
            image = Image.open(figures_path + f"/{col}.png")
            x = i // 2
            y = i % 2
            axes[x, y].imshow(np.array(image))
            axes[x, y].set_axis_off()
            os.remove(figures_path + f"/{col}.png")

        plt.subplots_adjust(wspace=0)
        plt.savefig(figures_path + f"/{name}.png", bbox_inches="tight")
