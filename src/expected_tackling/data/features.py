import numpy as np
import pandas as pd
import threading
from multiprocessing import Manager


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
    return (np.degrees(angle) + 360) % 360


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


def _compute_blockers_features(sample, frame_blockers, nb_blockers=3):
    frame_blockers = frame_blockers.copy()
    columns_to_suffix = ["s", "a", "dis", "o", "dir"]
    columns_suffix_dict = {col: col + "_blocker" for col in columns_to_suffix}

    frame_blockers["distance_to_blocker"] = frame_blockers.apply(
        lambda x: _compute_distance_between_players(sample["x"], sample["y"], x["x"], x["y"]), axis=1
    )
    frame_blockers["direction_to_blocker"] = frame_blockers.apply(
        lambda x: _compute_angle_between_players(sample["x"], sample["y"], x["x"], x["y"]), axis=1
    )
    frame_blockers = frame_blockers.sort_values(by="distance_to_blocker", ascending=True).reset_index()
    frame_blockers = frame_blockers[columns_to_suffix + ["distance_to_blocker", "direction_to_blocker"]].head(
        nb_blockers
    )

    frame_blockers = frame_blockers.rename(columns=columns_suffix_dict)

    blockers_features = frame_blockers.stack().rename(sample.name)
    blockers_features.index = blockers_features.index.map(lambda x: f"{x[1]}_{x[0]+1}")
    return blockers_features


def _compute_sample_features(sample, frame_ball_carrier, frame_blockers):
    sample = sample.copy()
    sample["distance_to_ball_carrier"] = _compute_distance_between_players(
        sample["x"], sample["y"], frame_ball_carrier["x"], frame_ball_carrier["y"]
    )
    sample["direction_to_ball_carrier"] = _compute_angle_between_players(
        sample["x"], sample["y"], frame_ball_carrier["x"], frame_ball_carrier["y"]
    )

    blockers_features = _compute_blockers_features(sample, frame_blockers)

    sample_features = pd.concat([sample, blockers_features])

    return sample_features


def _compute_group_features(group, frame_ball_carrier, frame_blockers):
    return group.apply(
        lambda x: _compute_sample_features(
            x,
            frame_ball_carrier,
            frame_blockers,
        ),
        axis=1,
    )


def _inverse_left_directed_plays(features_data):
    left_playDirection = features_data["playDirection"] == "left"
    blockers_columns = [
        col
        for col in features_data.columns
        if col.startswith("o_blocker") or col.startswith("dir_blocker") or col.startswith("direction_to_blocker")
    ]
    for col in ["o", "dir", "o_ball_carrier", "dir_ball_carrier", "direction_to_ball_carrier"] + blockers_columns:
        features_data.loc[left_playDirection, col] = (180 - features_data.loc[left_playDirection, col]) % 360

    return features_data


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
    ][["gameId", "playId", "nflId", "frameId", "x", "y", "playDirection", "s", "a", "dis", "o", "dir"]].set_index(
        ["gameId", "playId", "frameId", "nflId"]
    )

    features_data = (
        defense.groupby(["gameId", "playId", "frameId"])
        .apply(
            lambda x: _compute_group_features(
                x,
                ball_carrier.loc[(x["gameId"].iloc[0], x["playId"].iloc[0], x["frameId"].iloc[0])],
                blockers.loc[(x["gameId"].iloc[0], x["playId"].iloc[0], x["frameId"].iloc[0])],
            )
        )
        .reset_index(drop=True)
    )
    features_data = features_data[
        defense.columns.to_list() + [col for col in features_data.columns if col not in defense.columns]
    ]

    ball_carrier["ball_carrier_distance_to_sideline"] = ball_carrier["y"].apply(_compute_distance_to_nearest_sideline)
    ball_carrier["ball_carrier_distance_to_endzone"] = ball_carrier.apply(
        lambda x: _compute_distance_to_endzone(x["x"], x["playDirection"]), axis=1
    )

    features_data = features_data.merge(
        ball_carrier[
            ["s", "a", "dis", "o", "dir", "ball_carrier_distance_to_sideline", "ball_carrier_distance_to_endzone"]
        ],
        left_on=["gameId", "playId", "frameId"],
        right_index=True,
        suffixes=("", "_ball_carrier"),
    )

    features_data = _inverse_left_directed_plays(features_data)

    return features_data


def compute_features_data_with_threads(targeted_data, tracking, nb_threads=10):
    tracking_games = tracking["gameId"].unique()
    total_items = len(tracking_games)
    chunk_size = total_items // nb_threads

    manager = Manager()
    shared_dataframe_list = manager.list()

    def thread_function(i):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < nb_threads - 1 else total_items
        features_data = compute_features_data(
            targeted_data, tracking[tracking["gameId"].isin(tracking_games[start:end])]
        )

        shared_dataframe_list.append(features_data)

    threads = []
    for i in range(nb_threads):
        thread = threading.Thread(target=thread_function, args=(i,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    result_df = pd.concat(shared_dataframe_list, ignore_index=True)

    return result_df
