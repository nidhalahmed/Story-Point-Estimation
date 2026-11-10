import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path.cwd()

pairs = ["pair_25", "pair_17", "pair_05", "pair_36", "pair_04"]

for pair in pairs:
    metadata = pd.read_csv(
        ROOT / "data" / "processed" / "splits" /
        pair / "embeddings" / "development_metadata.csv"
    )

    y = pd.to_numeric(metadata["story_points"], errors="coerce")

    print(f"\n{'='*55}")
    print(pair)
    print("Rows:", len(y))
    print("Missing:", y.isna().sum())
    print("Unique labels:", y.nunique())
    print(y.describe(percentiles=[
        0.50, 0.75, 0.90, 0.95, 0.99, 0.999
    ]))

    print("\nLargest labels:")
    print(
        metadata.loc[y.nlargest(15).index]
        [["issue_id", "story_points"]]
        .to_string(index=False)
    )

from sklearn.model_selection import KFold

for pair in ["pair_25", "pair_17"]:
    metadata = pd.read_csv(
        ROOT / "data" / "processed" / "splits" /
        pair / "embeddings" / "development_metadata.csv"
    )

    y = metadata["story_points"].to_numpy()

    kf = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    print(f"\n{pair}")

    for fold, (train_idx, val_idx) in enumerate(
        kf.split(np.arange(len(y))), start=1
    ):
        y_val = y[val_idx]

        print(
            f"Fold {fold}: "
            f"train={len(train_idx)}, "
            f"validation={len(val_idx)}, "
            f"min={y_val.min()}, "
            f"median={np.median(y_val)}, "
            f"max={y_val.max()}, "
            f"MAE of median predictor="
            f"{np.mean(np.abs(y_val - np.median(y[train_idx]))):.4f}"
        )

        if y_val.max() > 100:
            print("  Potential extreme validation labels:")
            print(
                metadata.iloc[val_idx]
                .loc[lambda df: df["story_points"] > 100,
                     ["issue_id", "story_points"]]
                .to_string(index=False)
            )

            