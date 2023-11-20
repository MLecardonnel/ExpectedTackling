def _compute_group_features(group):
    group = group.sort_values("frameId")
    argmax = group["ott"].argmax()
    res = group[
        ["tackling_probability", "distance_to_ball_carrier", "ball_carrier_distance_to_endzone", "frameId"]
    ].iloc[argmax]
    res["frame_progression"] = (res["frameId"] - group["frameId"].min()) / (
        group["frameId"].max() - group["frameId"].min()
    )
    res["nb_frames_from_end"] = group["frameId"].max() - res["frameId"]
    res = res.drop(index="frameId")
    res["last_distance_to_ball_carrier"] = group.iloc[-1]["distance_to_ball_carrier"]
    res["last_tackling_probability"] = group.iloc[-1]["tackling_probability"]
    res["ball_carrier_distance_won_to_last_frame"] = max(
        res["ball_carrier_distance_to_endzone"] - group.iloc[-1]["ball_carrier_distance_to_endzone"], 0
    )
    res = res.drop(index="ball_carrier_distance_to_endzone")
    return res


def compute_mott_features_data(features_data, tackling_probability, tackles):
    features_data = features_data[
        ["gameId", "playId", "nflId", "frameId", "distance_to_ball_carrier", "ball_carrier_distance_to_endzone"]
    ].merge(tackling_probability, on=["gameId", "playId", "nflId", "frameId"])

    features_data["ott"] = features_data["tackling_probability"] / features_data["distance_to_ball_carrier"]

    mott_features_data = features_data.groupby(["gameId", "playId", "nflId"]).apply(_compute_group_features)

    tackles["tackle_or_assist"] = tackles[["tackle", "assist"]].max(axis=1)
    mott_features_data = mott_features_data.merge(
        tackles[(tackles["tackle_or_assist"] == 1)].set_index(["gameId", "playId", "nflId"])[["tackle_or_assist"]],
        how="left",
        left_index=True,
        right_index=True,
    )
    mott_features_data["tackle_or_assist"] = mott_features_data["tackle_or_assist"].fillna(0)

    mott_features_data = mott_features_data.merge(
        tackles[tackles["pff_missedTackle"] == 1].set_index(["gameId", "playId", "nflId"])[["pff_missedTackle"]],
        how="left",
        left_index=True,
        right_index=True,
    )
    mott_features_data["pff_missedTackle"] = mott_features_data["pff_missedTackle"].fillna(0)

    return mott_features_data
