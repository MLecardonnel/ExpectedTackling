import numpy as np


def create_target(visualization_tracking_data, tackles):
    tackles = tackles.copy()
    tackles["tackle_or_assist"] = tackles[["tackle", "assist"]].max(axis=1)
    tackles = tackles[(tackles["tackle_or_assist"] == 1)][["gameId", "playId", "nflId", "tackle_or_assist"]]
    targeted_data = visualization_tracking_data.merge(tackles, how="left", on=["gameId", "playId", "nflId"])

    targeted_data["will_tackle"] = np.nan
    targeted_data.loc[~targeted_data["ball_carrier_id"].isna(), "will_tackle"] = 0
    targeted_data.loc[
        (targeted_data["ballCarrierId"] == targeted_data["ball_carrier_id"]) & (targeted_data["tackle_or_assist"] == 1),
        "will_tackle",
    ] = 1
    return targeted_data
