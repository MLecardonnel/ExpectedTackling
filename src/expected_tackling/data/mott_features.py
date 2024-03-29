import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def _find_peaks(ott: pd.Series) -> list:
    peaks = find_peaks(ott.tolist() + [0], height=0.5, distance=16)[0].tolist()
    if len(peaks) == 0:
        peaks = [ott.argmax()]
    return peaks


def _compute_peak_features(group: pd.DataFrame, peak: int) -> pd.Series:
    res = group[["frameId", "ott", "ball_carrier_distance_to_endzone"]].iloc[peak]
    res["mean_distance_to_ball_carrier_from_peak"] = group.iloc[peak:]["distance_to_ball_carrier"].mean()
    res["ball_carrier_distance_won_to_last_frame"] = max(
        res["ball_carrier_distance_to_endzone"] - group.iloc[-1]["ball_carrier_distance_to_endzone"], 0
    )
    res = res.drop(index="ball_carrier_distance_to_endzone")
    return res


def _compute_group_features(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("frameId")
    peaks = _find_peaks(group["ott"])
    res = pd.concat([_compute_peak_features(group, peak) for peak in peaks], axis=1)
    res = res.T.reset_index(drop=True)
    res.index.name = "opportunityId"
    res = res.set_index("frameId", append=True)
    return res


def compute_mott_features_data(
    features_data: pd.DataFrame, tackling_probability: pd.DataFrame, tackles: pd.DataFrame
) -> pd.DataFrame:
    """Compute MOTT (Missed Opportunities To Tackle) features for every identified tackling
    opportunities for each defensive player.

    Parameters
    ----------
    features_data : pd.DataFrame
        DataFrame with computed movement features for defensive players.
    tackling_probability : pd.DataFrame
        DataFrame containing tackling probabilities.
    tackles : pd.DataFrame
        DataFrame containing information about tackles and assists.

    Returns
    -------
    pd.DataFrame
        DataFrame with MOTT features.
    """
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
    mott_features_data.loc[mott_features_data.index.droplevel([3, 4]).duplicated(keep="last"), "tackle_or_assist"] = 0

    mott_features_data = mott_features_data.merge(
        tackles[tackles["pff_missedTackle"] == 1].set_index(["gameId", "playId", "nflId"])[["pff_missedTackle"]],
        how="left",
        left_index=True,
        right_index=True,
    )
    mott_features_data["pff_missedTackle"] = mott_features_data["pff_missedTackle"].fillna(0)
    mott_features_data.loc[
        (mott_features_data.index.droplevel([3, 4]).duplicated(keep=False))
        & (~mott_features_data.index.droplevel([3, 4]).duplicated(keep="last"))
        & (mott_features_data["tackle_or_assist"] == 1),
        "pff_missedTackle",
    ] = 0

    return mott_features_data


def sample_training_data(mott_features_data: pd.DataFrame, negatives_multplier: int = 10) -> pd.DataFrame:
    """Sample training data for MOTT (Missed Opportunities To Tackle) features.

    Parameters
    ----------
    mott_features_data : pd.DataFrame
        DataFrame containing MOTT features.
    negatives_multplier : int, optional
        Multiplier for the number of negative samples, by default 10.

    Returns
    -------
    pd.DataFrame
        Sampled DataFrame with training data for MOTT features.
    """
    mott_features_data_for_sampling = mott_features_data[
        ~mott_features_data.index.droplevel([3, 4]).duplicated(keep=False)
    ].copy()
    mott_features_data_for_sampling = mott_features_data_for_sampling[mott_features_data_for_sampling["ott"] != np.inf]

    sample_mott_features_data = pd.concat(
        [
            mott_features_data_for_sampling[
                (mott_features_data_for_sampling["pff_missedTackle"] == 1)
                & (mott_features_data_for_sampling["ott"] > 0.1)
            ],
            mott_features_data_for_sampling[mott_features_data_for_sampling["pff_missedTackle"] == 0].sample(
                len(mott_features_data_for_sampling[mott_features_data_for_sampling["pff_missedTackle"] == 1])
                * negatives_multplier,
                random_state=42,
            ),
        ]
    )
    return sample_mott_features_data
