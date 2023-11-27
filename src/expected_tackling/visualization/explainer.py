import io
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from shapash import SmartExplainer

figures_path = str(Path(__file__).parents[3] / "reports/figures")


class Explainer:
    def __init__(self, X, y, model, sample_size=100000):
        if X.shape[0] > sample_size:
            X = X.sample(sample_size)

        xpl = SmartExplainer(
            model=model,
        )

        xpl.compile(
            x=X,
            y_target=y.loc[X.index],
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
            contribution_plot = self.xpl.plot.contribution_plot(col=col)
            image = Image.open(io.BytesIO(contribution_plot.to_image(format="png")))
            x = i // 2
            y = i % 2
            axes[x, y].imshow(np.array(image))
            axes[x, y].set_axis_off()

        plt.subplots_adjust(wspace=0)
        plt.savefig(figures_path + f"/{name}.png", bbox_inches="tight")
