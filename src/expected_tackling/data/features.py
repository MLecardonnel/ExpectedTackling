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


def _compute_distance_between_players(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def _compute_angle_between_players(x1, y1, x2, y2):
    angle = np.arctan2(x2 - x1, y2 - y1)
    return np.degrees(angle)


def _compute_distance_to_nearest_sideline(y, field_width=53.3):
    return min(y - 0, field_width - y)


def _compute_distance_to_endzone(x, playDirection, field_length=120):
    field_subdivision = field_length / 12
    if playDirection == "right":
        distance = field_length - field_subdivision - x
    elif playDirection == "left":
        distance = x - field_subdivision
    else:
        raise ValueError
    return distance


def _compute_sample_features(sample, ball_carrier):
    frame_ball_carrier = ball_carrier.loc[(sample["gameId"], sample["playId"], sample["frameId"])]

    sample["distance_to_ball_carrier"] = _compute_distance_between_players(
        sample["x"], sample["y"], frame_ball_carrier["x"], frame_ball_carrier["y"]
    )
    sample["direction_to_ball_carrier"] = _compute_angle_between_players(
        sample["x"], sample["y"], frame_ball_carrier["x"], frame_ball_carrier["y"]
    )

    return sample


def compute_features_data(targeted_data, tracking):
    merged_data = targeted_data.merge(
        tracking[["gameId", "playId", "nflId", "frameId"] + [col for col in tracking if col not in targeted_data]],
        on=["gameId", "playId", "nflId", "frameId"],
    )

    defense = merged_data[(merged_data["is_defense"]) & (~merged_data["ball_carrier_id"].isna())][
        ["gameId", "playId", "nflId", "frameId", "x", "y", "playDirection", "will_tackle", "s", "a", "dis", "o", "dir"]
    ].reset_index(drop=True)

    ball_carrier = merged_data[
        (~merged_data["is_defense"]) & (~merged_data["ball_carrier_id"].isna()) & (merged_data["is_ball_carrying"])
    ][["gameId", "playId", "nflId", "frameId", "x", "y", "playDirection", "s", "a", "dis", "o", "dir"]].set_index(
        ["gameId", "playId", "frameId"]
    )

    blockers = merged_data[
        (~merged_data["is_defense"]) & (~merged_data["ball_carrier_id"].isna()) & (~merged_data["is_ball_carrying"])
    ][["gameId", "playId", "nflId", "frameId", "x", "y", "playDirection", "s", "a", "dis", "o", "dir"]].reset_index(
        drop=True
    )

    features_data = defense.apply(lambda x: _compute_sample_features(x, ball_carrier), axis=1)

    ball_carrier_distance_to_sideline = (
        ball_carrier["y"].apply(_compute_distance_to_nearest_sideline).rename("ball_carrier_distance_to_sideline")
    )
    ball_carrier_distance_to_endzone = ball_carrier.apply(
        lambda x: _compute_distance_to_endzone(x["x"], x["playDirection"]), axis=1
    ).rename("ball_carrier_distance_to_endzone")

    features_data = features_data.merge(
        ball_carrier_distance_to_sideline,
        left_on=["gameId", "playId", "frameId"],
        right_index=True,
    )

    features_data = features_data.merge(
        ball_carrier_distance_to_endzone,
        left_on=["gameId", "playId", "frameId"],
        right_index=True,
    )

    return features_data
