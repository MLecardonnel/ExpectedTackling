import copy
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.cm import Reds
from matplotlib.colors import to_hex

animations_path = str(Path(__file__).parents[3] / "reports/animations")


class Field:
    """Class for visualizing a football field with tracking data for a play"""

    def __init__(self, field_width: float = 53.3, field_length: float = 120.0, step_duration: int = 50):
        """Initialize the Field object.

        Parameters
        ----------
        field_width : float, optional
            Width of the football field, by default 53.3
        field_length : float, optional
            Length of the football field, by default 120.0
        step_duration : int, optional
            Duration for animation steps, by default 50
        """
        self.field_width = field_width
        self.field_length = field_length
        self.field_subdivision = field_length / 12

        self.step_duration = step_duration

        self.color_field = "#96B78C"
        self.color_endzone = "#6F976D"
        self.color_lines = "white"

        fig = go.Figure()

        fig = self._draw_numbers_on_field(fig, field_width - 5)
        fig = self._draw_numbers_on_field(fig, 5)

        fig = self._draw_rectangle_on_field(fig, x0=0, y0=0, x1=field_length, y1=field_width, color=self.color_field)
        fig = self._draw_rectangle_on_field(
            fig, x0=0, y0=0, x1=self.field_subdivision, y1=field_width, color=self.color_endzone
        )
        fig = self._draw_rectangle_on_field(
            fig,
            x0=field_length - self.field_subdivision,
            y0=0,
            x1=field_length,
            y1=field_width,
            color=self.color_endzone,
        )

        for i in range(2, 23):
            fig = self._draw_line_on_field(
                fig,
                x=i * self.field_subdivision / 2,
                color=self.color_lines,
                width=2 if i % 2 == 0 else 1,
            )

        fig.update_layout(
            xaxis={"range": [-5, field_length + 5], "visible": False},
            yaxis={"range": [-5, field_width + 5], "visible": False, "scaleanchor": "x", "scaleratio": 1},
            height=600,
            updatemenus=[
                {
                    "buttons": [
                        {
                            "args": [
                                None,
                                {
                                    "frame": {"duration": self.step_duration, "redraw": False},
                                    "fromcurrent": True,
                                    "transition": {"duration": 0},
                                },
                            ],
                            "label": "⏵",
                            "method": "animate",
                        },
                        {
                            "args": [
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                },
                            ],
                            "label": "⏸",
                            "method": "animate",
                        },
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 87},
                    "showactive": False,
                    "type": "buttons",
                    "x": 0.1,
                    "xanchor": "right",
                    "y": 0,
                    "yanchor": "top",
                }
            ],
            sliders=[
                {
                    "active": 0,
                    "yanchor": "top",
                    "xanchor": "left",
                    "currentvalue": {"font": {"size": 20}, "prefix": "frameId: ", "visible": True, "xanchor": "right"},
                    "transition": {"duration": self.step_duration, "easing": "cubic-in-out"},
                    "pad": {"b": 10, "t": 50},
                    "len": 0.9,
                    "x": 0.1,
                    "y": 0,
                    "steps": [],
                }
            ],
        )

        self.fig = fig

    def _draw_numbers_on_field(self, fig: go.Figure, y: float) -> go.Figure:
        numbers_on_field = ["10", "20", "30", "40", "50", "40", "30", "20", "10"]

        for i in range(len(numbers_on_field)):
            fig.add_annotation(
                x=(i + 2) * self.field_subdivision,
                y=y,
                text=numbers_on_field[i],
                showarrow=False,
                font={
                    "size": 25,
                    "color": self.color_lines,
                },
            )

        return fig

    def _draw_rectangle_on_field(
        self, fig: go.Figure, x0: float, y0: float, x1: float, y1: float, color: str
    ) -> go.Figure:
        fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1, line_width=0, layer="below", fillcolor=color)
        return fig

    def _draw_line_on_field(self, fig: go.Figure, x: float, color: str, width: int) -> go.Figure:
        fig.add_shape(
            type="line",
            x0=x,
            y0=0,
            x1=x,
            y1=self.field_width,
            layer="below",
            line={
                "color": color,
                "width": width,
            },
        )
        return fig

    def _create_step(self, frame_id: str) -> dict:
        step = {
            "args": [
                [frame_id],
                {
                    "frame": {"duration": self.step_duration, "redraw": False},
                    "mode": "immediate",
                    "transition": {"duration": 0},
                },
            ],
            "label": frame_id,
            "method": "animate",
        }
        return step

    def draw_scrimmage_and_first_down(self, absoluteYardlineNumber: int, yardsToGo: int, playDirection: str) -> None:
        """Draw scrimmage line and first down marker on the football field.

        Parameters
        ----------
        absoluteYardlineNumber : int
            Absolute yardline number on the football field
        yardsToGo : int
            Yards to go for a first down
        playDirection : str
            Direction of play
        """
        if playDirection == "right":
            yard_line_first_down = absoluteYardlineNumber + yardsToGo
        elif playDirection == "left":
            yard_line_first_down = absoluteYardlineNumber - yardsToGo
        else:
            raise ValueError

        self.fig = self._draw_line_on_field(self.fig, absoluteYardlineNumber, "#0070C0", 2)
        self.fig = self._draw_line_on_field(self.fig, yard_line_first_down, "#E9D11F", 2)

    def create_animation(self, play_tracking: pd.DataFrame) -> None:
        """Create an animation for player movements during a play.

        Parameters
        ----------
        play_tracking : pd.DataFrame
            DataFrame containing tracking data for players during the play.
        """
        frames = []
        steps = []
        for frame_id in play_tracking["frameId"].unique():
            frame_tracking = play_tracking[play_tracking["frameId"] == frame_id]
            players_tracking = frame_tracking[~frame_tracking["nflId"].isna()]

            ball_carrying_tracking = players_tracking[players_tracking["is_ball_carrying"]]
            defense_tracking = players_tracking[
                (~players_tracking["is_ball_carrying"]) & (players_tracking["is_defense"])
            ]
            offense_tracking = players_tracking[
                (~players_tracking["is_ball_carrying"]) & (~players_tracking["is_defense"])
            ]

            data = []

            data.append(
                go.Scatter(
                    x=offense_tracking["x"],
                    y=offense_tracking["y"],
                    mode="markers",
                    marker={"size": 10, "color": "black"},
                    name="offense",
                    hoverinfo="none",
                ),
            )

            if len(ball_carrying_tracking) != 0:
                data.append(
                    go.Scatter(
                        x=ball_carrying_tracking["x"],
                        y=ball_carrying_tracking["y"],
                        mode="markers",
                        marker={"size": 10, "color": "yellow", "opacity": 1},
                        name="ball_carrier",
                        hoverinfo="none",
                    ),
                )
            else:
                data.append(
                    go.Scatter(
                        x=[0],
                        y=[0],
                        mode="markers",
                        marker={"size": 10, "color": "yellow", "opacity": 0},
                        name="ball_carrier",
                        hoverinfo="none",
                    ),
                )

            data.append(
                go.Scatter(
                    x=defense_tracking["x"],
                    y=defense_tracking["y"],
                    mode="markers",
                    marker={"size": 10, "color": "white"},
                    name="defense",
                    hoverinfo="none",
                ),
            )

            steps.append(self._create_step(str(frame_id)))
            frames.append(
                {
                    "data": data,
                    "name": str(frame_id),
                }
            )
        for trace in frames[0]["data"]:
            self.fig.add_trace(trace)
        self.fig.frames = frames
        self.fig.layout.sliders[0]["steps"] = steps

    def _get_color(self, value: float) -> str:
        return to_hex(Reds(value))

    def create_tackling_probability_animation(self, play_tracking: pd.DataFrame, plot_mott: bool = False) -> None:
        """Create an animation for player movements during a play with tackling probability visualization.

        Parameters
        ----------
        play_tracking : pd.DataFrame
            DataFrame containing tracking data for players during the play.
        plot_mott : bool, optional
            Flag to plot MOTT (Missed Opportunity to Tackle) predictions, by default False.
        """
        frames = []
        steps = []
        mott_predictions = pd.DataFrame()
        for frame_id in play_tracking["frameId"].unique():
            frame_tracking = play_tracking[play_tracking["frameId"] == frame_id]
            players_tracking = frame_tracking[~frame_tracking["nflId"].isna()]

            ball_carrying_tracking = players_tracking[players_tracking["is_ball_carrying"]]
            defense_tracking = players_tracking[
                (~players_tracking["is_ball_carrying"]) & (players_tracking["is_defense"])
            ]
            offense_tracking = players_tracking[
                (~players_tracking["is_ball_carrying"]) & (~players_tracking["is_defense"])
            ]

            data = []

            if plot_mott:
                assert "mott" in play_tracking.columns
                mott_predictions = pd.concat([mott_predictions, frame_tracking[frame_tracking["mott"] == 1]])
                if len(mott_predictions) != 0:
                    data.append(
                        go.Scatter(
                            x=mott_predictions["x"],
                            y=mott_predictions["y"],
                            mode="markers",
                            marker={"size": 12, "color": "#FE962F", "opacity": 1, "symbol": "x", "line_width": 1},
                            customdata=np.stack(
                                (
                                    mott_predictions["nflId"].astype(int),
                                    mott_predictions["displayName"],
                                    mott_predictions["position"],
                                    mott_predictions["club"],
                                    mott_predictions["tackling_probability"].round(2),
                                ),
                                axis=-1,
                            ),
                            hovertemplate="<b>nflId:</b> %{customdata[0]}<br>"
                            "<b>Player:</b> %{customdata[1]} %{customdata[2]}<br>"
                            "<b>Team:</b> %{customdata[3]}<br>"
                            "<br>"
                            "<b>Tackling Probability:</b> %{customdata[4]}",
                            name="MOTT",
                        ),
                    )
                else:
                    data.append(
                        go.Scatter(
                            x=[0],
                            y=[0],
                            mode="markers",
                            marker={"size": 12, "color": "#FE962F", "opacity": 0, "symbol": "x", "line_width": 1},
                            name="MOTT",
                            hoverinfo="none",
                        ),
                    )

            data.append(
                go.Scatter(
                    x=offense_tracking["x"],
                    y=offense_tracking["y"],
                    mode="markers",
                    marker={"size": 10, "color": "black"},
                    name="offense",
                    hoverinfo="none",
                ),
            )

            if len(ball_carrying_tracking) != 0:
                data.append(
                    go.Scatter(
                        x=ball_carrying_tracking["x"],
                        y=ball_carrying_tracking["y"],
                        mode="markers",
                        marker={"size": 10, "color": "yellow", "opacity": 1},
                        name="ball_carrier",
                        hoverinfo="none",
                    ),
                )
            else:
                data.append(
                    go.Scatter(
                        x=[0],
                        y=[0],
                        mode="markers",
                        marker={"size": 10, "color": "yellow", "opacity": 0},
                        name="ball_carrier",
                        hoverinfo="none",
                    ),
                )

            data.append(
                go.Scatter(
                    x=defense_tracking["x"],
                    y=defense_tracking["y"],
                    mode="markers",
                    marker={
                        "size": 10,
                        "color": defense_tracking["tackling_probability"].apply(self._get_color),
                        "cmin": 0,
                        "cmax": 1,
                        "colorscale": "Reds",
                        "colorbar": dict(title="defense", thickness=10, x=1.03, y=0.4, len=0.85),
                    },
                    customdata=np.stack(
                        (
                            defense_tracking["nflId"].astype(int),
                            defense_tracking["displayName"],
                            defense_tracking["position"],
                            defense_tracking["club"],
                            defense_tracking["tackling_probability"].round(2),
                        ),
                        axis=-1,
                    ),
                    hovertemplate="<b>nflId:</b> %{customdata[0]}<br>"
                    "<b>Player:</b> %{customdata[1]} %{customdata[2]}<br>"
                    "<b>Team:</b> %{customdata[3]}<br>"
                    "<br>"
                    "<b>Tackling Probability:</b> %{customdata[4]}",
                    name="defense",
                    showlegend=False,
                ),
            )

            steps.append(self._create_step(str(frame_id)))
            frames.append(
                {
                    "data": data,
                    "name": str(frame_id),
                }
            )
        for trace in frames[0]["data"]:
            self.fig.add_trace(trace)
        self.fig.frames = frames
        self.fig.layout.sliders[0]["steps"] = steps

    def create_mott_predictions_animation(self, play_tracking: pd.DataFrame, mott_predictions: pd.DataFrame) -> None:
        """Create an animation for player movements during a football play with highlighted MOTT predictions.

        Parameters
        ----------
        play_tracking : pd.DataFrame
            DataFrame containing tracking data for players during the play.
        mott_predictions : pd.DataFrame
            DataFrame containing MOTT predictions for players.
        """
        mott_predictions = mott_predictions[mott_predictions["mott"] == 1].sort_values("frameId")
        motts = [
            f"{record['position']} {record['displayName']} ({int(record['nflId'])}) on frame {int(record['frameId'])}"
            for record in mott_predictions.to_dict("records")
        ]
        if len(motts) > 0:
            text = "MOTT: "
            for i in range(int(np.ceil(len(motts) / 3))):
                text += ", ".join(motts[3 * i : 3 * (i + 1)]) + "<br>"
            self.fig.layout.title = text

        play_tracking = play_tracking.merge(
            mott_predictions[["gameId", "playId", "nflId", "frameId", "mott"]],
            how="left",
            on=["gameId", "playId", "nflId", "frameId"],
        )

        self.create_tackling_probability_animation(play_tracking, plot_mott=True)

    def save_as_gif(self, name: str = "animated_play") -> None:
        """Save the animated play as a GIF file.

        Parameters
        ----------
        name : str, optional
            Name of the saved GIF file, by default "animated_play"
        """
        layout = copy.deepcopy(self.fig.layout)
        with imageio.get_writer(animations_path + f"/{name}.gif", mode="I", loop=0) as writer:
            for i, frame in enumerate(self.fig.frames):
                layout.sliders[0]["active"] = i
                fig = go.Figure(frame.data, layout=layout)
                writer.append_data(imageio.imread(fig.to_image(format="png", height=600, width=1350)))
