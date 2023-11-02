import pandas as pd


POSSIBLE_LAST_EVENT = ["tackle", "out_of_bounds", "touchdown", "fumble", "qb_slide", "safety"]
RUN_EVENT = ["handoff", "run"]
BALL_SNAP_EVENT = ["ball_snap", "snap_direct", "autoevent_ballsnap"]


def get_valid_plays_from_events(tracking):
    plays_frames = tracking.drop_duplicates(["gameId", "playId", "frameId"])[["gameId", "playId", "frameId", "event"]]
    plays_events = plays_frames.dropna(subset=["event"]).groupby(["gameId", "playId"])["event"].unique()

    events_sequences_counts = plays_events.astype(str).value_counts()
    events_sequences = pd.Series(events_sequences_counts[events_sequences_counts > 1].index)
    events_sequences = events_sequences[
        events_sequences.apply(
            lambda x: "pass_outcome_caught" not in x or ("pass_outcome_caught" in x and "pass_arrived" in x)
        )
    ]

    valid_plays = plays_events[plays_events.astype(str).isin(events_sequences.values)]

    plays_frames_valid = plays_frames.set_index(["gameId", "playId"])
    plays_frames_valid = plays_frames_valid.loc[valid_plays.index]

    return plays_frames_valid, plays_events


def _get_ball_carrier_from_event(group, plays_events):
    events = plays_events.loc[(group["gameId"].iloc[0], group["playId"].iloc[0])]
    if any(element in events for element in RUN_EVENT):
        is_RUN_EVENT = group["event"].isin(RUN_EVENT)
        group.loc[is_RUN_EVENT, "ball_carrier"] = "ball_carrier"
        if any(element in events for element in POSSIBLE_LAST_EVENT):
            group.loc[group["event"].isin(POSSIBLE_LAST_EVENT), "ball_carrier"] = "ball_carrier"

        if is_RUN_EVENT.idxmax() != 0:
            group.loc[is_RUN_EVENT.idxmax() - 1, "ball_carrier"] = "qb"

        if any(element in events for element in BALL_SNAP_EVENT):
            group.loc[group["event"].isin(BALL_SNAP_EVENT), "ball_carrier"] = "qb"

            valid_indices = group["ball_carrier"].notnull()
            start_index = valid_indices.idxmax()
            end_index = valid_indices[::-1].idxmax()
            group.loc[start_index:end_index, "ball_carrier"] = (
                group.loc[start_index:end_index, "ball_carrier"].ffill().bfill()
            )

        else:
            group["ball_carrier"] = group["ball_carrier"].bfill()

    elif "pass_arrived" in events:
        is_pass_event = group["event"] == "pass_arrived"
        group.loc[is_pass_event, "ball_carrier"] = "ball_carrier"
        if any(element in events for element in POSSIBLE_LAST_EVENT):
            group.loc[group["event"].isin(POSSIBLE_LAST_EVENT), "ball_carrier"] = "ball_carrier"

        valid_indices = group["ball_carrier"].notnull()
        start_index = valid_indices.idxmax()
        end_index = valid_indices[::-1].idxmax()
        group.loc[start_index:end_index, "ball_carrier"] = (
            group.loc[start_index:end_index, "ball_carrier"].ffill().bfill()
        )

    return group


def _get_ball_carrier_id(x):
    if x["ball_carrier"] == "qb":
        res = x["qbId"]
    elif x["ball_carrier"] == "ball_carrier":
        res = x["ballCarrierId"]
    else:
        res = None
    return res


def compute_visualization_data(plays_frames_valid, plays_events, plays, players, tracking):
    visualization_data = plays_frames_valid.reset_index()
    visualization_data["ball_carrier"] = None

    visualization_data = (
        visualization_data.groupby(["gameId", "playId"])
        .apply(lambda x: _get_ball_carrier_from_event(x, plays_events))
        .reset_index(drop=True)
    )

    visualization_data = visualization_data.merge(
        plays[["gameId", "playId", "ballCarrierId", "defensiveTeam", "absoluteYardlineNumber", "yardsToGo"]],
        on=["gameId", "playId"],
    )

    players_positions = (
        tracking[["gameId", "playId", "nflId"]]
        .drop_duplicates()
        .dropna()
        .merge(players[["nflId", "position"]], on="nflId")
    )
    qb_players = (
        players_positions[players_positions["position"] == "QB"]
        .drop_duplicates(subset=["gameId", "playId"])
        .drop(columns="position")
        .rename(columns={"nflId": "qbId"})
    )
    visualization_data = visualization_data.merge(qb_players, on=["gameId", "playId"])

    visualization_data["ball_carrier_id"] = visualization_data.apply(_get_ball_carrier_id, axis=1)

    visualization_tracking_data = tracking[
        ["gameId", "playId", "nflId", "frameId", "club", "x", "y", "playDirection"]
    ].merge(visualization_data, how="inner", on=["gameId", "playId", "frameId"])
    visualization_tracking_data["is_defense"] = (
        visualization_tracking_data["club"] == visualization_tracking_data["defensiveTeam"]
    )
    visualization_tracking_data["is_ball_carrying"] = (
        visualization_tracking_data["nflId"] == visualization_tracking_data["ball_carrier_id"]
    )

    return visualization_tracking_data
