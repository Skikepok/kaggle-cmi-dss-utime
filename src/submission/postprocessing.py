import torch

from src.config import FT_COLUMNS

MIN_WINDOW_TO_STEPS = 30 * 60 / 5  # Nb of steps of 5s in 30 minutes
MATCH_LIMIT = 8 * 3600 / 5  # Number of steps in 8h


def add_predictions(df, model):
    X = torch.from_numpy(df[FT_COLUMNS].values).unsqueeze(0)

    predictions = model(X)

    df["pred_sleeping"] = (predictions[0, 1, :] >= 0.9) * 1

    return df


def labelize_event(df):
    df.loc[df.iloc[[0, -1]].index, "pred_sleeping"] = 0

    states_changes = df.pred_sleeping.diff()[1:]
    states_changes = states_changes[states_changes != 0]

    assert states_changes.iloc[0] == 1

    nb_events = states_changes.shape[0]

    assert nb_events % 2 == 0

    index = 0
    df["predicted_labels"] = 0
    tmp_sleep_number = 1
    df["tmp_sleep_number"] = 0

    while index + 1 != nb_events:
        onset = states_changes.index[index]

        while True:
            index += 1
            wakeup = states_changes.index[index]

            if index + 1 == nb_events:
                break

            potential_onset = states_changes.index[index + 1]

            diff = potential_onset - wakeup

            index += 1

            if diff <= MIN_WINDOW_TO_STEPS:
                continue
            else:
                break

        sleep_time = wakeup - onset

        if sleep_time >= MIN_WINDOW_TO_STEPS:
            df.loc[onset, "predicted_labels"] = 1
            df.loc[wakeup, "predicted_labels"] = -1
            df.loc[[onset, wakeup], "tmp_sleep_number"] = tmp_sleep_number
            tmp_sleep_number += 1

    return df


def select_best_night_candidates(df):
    nights_lenghts = (
        df[df.tmp_sleep_number > 0]
        .groupby("tmp_sleep_number")
        .apply(lambda x: x.iloc[1].step - x.iloc[0].step)
        .to_dict()
    )

    midnights = df[df["timestamp"].map(lambda x: str(x)[-13:-5] == "00:00:00")]

    to_keep = []

    for _, midnight in midnights.iterrows():
        possible_nights = df.loc[
            (df.predicted_labels == 1)
            & (abs(df.step.astype(float) - midnight.step) < MATCH_LIMIT)
        ]

        if possible_nights.shape[0] == 0:
            continue

        max_night = possible_nights.loc[
            possible_nights.tmp_sleep_number.map(nights_lenghts).idxmax()
        ]

        to_keep.append(max_night.tmp_sleep_number)

    df.loc[~df["tmp_sleep_number"].isin(to_keep), "predicted_labels"] = 0

    return df


def make_submission_df(df):
    submission = df.loc[df.predicted_labels != 0].copy().reset_index(drop=True)
    submission["event"] = submission.predicted_labels.map({1: "onset", -1: "wakeup"})
    submission["score"] = 1.0
    submission.index.name = "row_id"

    return submission[
        [
            "series_id",
            "step",
            "event",
            "score",
        ]
    ]
