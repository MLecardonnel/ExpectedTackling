import numpy as np
import plotly.graph_objects as go


class Field:
    def __init__(self, field_width=53.3, field_length=120):
        self.field_width = field_width
        self.field_length = field_length
        self.field_subdivision = field_length / 12

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
            yaxis={"range": [-5, field_width + 5], "visible": False},
            updatemenus=[
                {
                    "buttons": [
                        {
                            "args": [
                                None,
                                {
                                    "frame": {"duration": 200, "redraw": False},
                                    "fromcurrent": True,
                                    "transition": {"duration": 0},
                                },
                            ],
                            "label": "Play",
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
                            "label": "Pause",
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
                    "transition": {"duration": 200, "easing": "cubic-in-out"},
                    "pad": {"b": 10, "t": 50},
                    "len": 0.9,
                    "x": 0.1,
                    "y": 0,
                    "steps": [],
                }
            ],
        )

        self.fig = fig

    def _draw_numbers_on_field(self, fig, y):
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

    def _draw_rectangle_on_field(self, fig, x0, y0, x1, y1, color):
        fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1, line_width=0, layer="below", fillcolor=color)
        return fig

    def _draw_line_on_field(self, fig, x, color, width):
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

    def draw_scrimmage_and_first_down(self, absoluteYardlineNumber, yardsToGo, playDirection):
        if playDirection == "right":
            yard_line_first_down = absoluteYardlineNumber + yardsToGo
        elif playDirection == "left":
            yard_line_first_down = absoluteYardlineNumber - yardsToGo
        else:
            raise ValueError

        self.fig = self._draw_line_on_field(self.fig, absoluteYardlineNumber, "#0070C0", 2)
        self.fig = self._draw_line_on_field(self.fig, yard_line_first_down, "#E9D11F", 2)
