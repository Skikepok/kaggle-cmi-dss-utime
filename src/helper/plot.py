from plotly.subplots import make_subplots
import plotly.graph_objects as go

import matplotlib.pyplot as plt
import numpy as np


def plot(df, series_id=None, start=0, end=0, interactive=True):
    if series_id is None:
        series_id = np.random.choice(df.series_id.unique())

    if not end:
        df = df[df["series_id"] == series_id].copy().reset_index(drop=True)
    else:
        df = (
            df[df["series_id"] == series_id]
            .copy()
            .reset_index(drop=True)
            .iloc[start:end]
        )

    midnights = df[df["timestamp"].map(lambda x: str(x)[-13:-5] == "00:00:00")]

    fig = make_subplots(rows=1, cols=1, subplot_titles=["anglez", "enmo"])
    fig.update_layout(height=400, width=1200, title_text=f"Event N°{series_id}")

    X_col = "step"

    for i, col in enumerate(["anglez", "enmo"]):
        fig.add_trace(
            go.Scatter(
                x=df[X_col],
                y=df[col] / df[col].max(),
                mode="lines",
                line=dict(
                    color=(
                        "rgba(39, 114, 245, 0.5)"
                        if col == "anglez"
                        else "rgba(0, 0, 0, 0.5)"
                    )
                ),
                name=col,
            ),
            row=1,
            col=1,
        )

    if "sleeping" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df[X_col],
                y=df["sleeping"],
                mode="lines",
                line=dict(color="rgba(186, 23, 23, 0.5)"),
                name="Sleeping",
            ),
            row=1,
            col=1,
        )

    if "pred_sleeping" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df[X_col],
                y=df["pred_sleeping"],
                mode="lines",
                line=dict(color="rgba(6, 185, 0, 0.5)"),
                name="predictions",
            ),
            row=1,
            col=1,
        )

    if "predicted_labels" in df.columns:
        for _, row in df.loc[df.predicted_labels == 1].iterrows():
            fig.add_vline(
                x=row.step, line=dict(color="rgba(245, 183, 39, 0.80)", dash="dot")
            )

        for _, row in df.loc[df.predicted_labels == -1].iterrows():
            fig.add_vline(
                x=row.step, line=dict(color="rgba(209, 0, 0, 0.80)", dash="dot")
            )

    for _, row in midnights.iterrows():
        fig.add_vline(
            x=row.step,
            line=dict(color="rgba(39, 196, 245, 0.75)", dash="dot"),
            name="Midnight",
        )

    # Add labels for better visibility
    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[0],
            mode="lines",
            line=dict(color="rgba(39, 196, 245, 0.75)", dash="dot"),
            name="Midnight",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[0],
            mode="lines",
            line=dict(color="rgba(245, 183, 39, 0.80)", dash="dot"),
            name="Onset prediction",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[0],
            mode="lines",
            line=dict(color="rgba(209, 0, 0, 0.80)", dash="dot"),
            name="Wakeup prediction",
        ),
        row=1,
        col=1,
    )

    fig.show(config={"staticPlot": not interactive})
