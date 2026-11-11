import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, median_absolute_error

# 1. CONFIG

ROOT = Path(__file__).resolve().parent.parent

PAIRS_FILE = ROOT / "config" / "project_pairs.csv"
MODELS_DIR = ROOT / "models"
SPLITS_DIR = ROOT / "data" / "processed" / "splits"

OUTPUT_DIR = ROOT / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_DIM = 384

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 65)
print("MEMBER 5: CROSS-PROJECT EVALUATION")
print("=" * 65)
print(f"Repository: {ROOT}")
print(f"Device: {DEVICE}")
print(f"Output directory: {OUTPUT_DIR}")

# 2. CNN, exact as Niraj

class StoryPointCNN(nn.Module):
    def __init__(self, input_dim=384, dropout=0.2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 8, kernel_size=3, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.regressor = nn.Sequential(
            nn.Linear(8 * (input_dim // 8), 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.features(x)
        x = x.flatten(start_dim=1)

        return self.regressor(x).squeeze(-1)


# helper functions, load, metric_calc and predict

def load_model(model_path):
    model = StoryPointCNN().to(DEVICE)

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(checkpoint)

    preprocess_path = model_path.parent / "preprocess.json"

    with open(preprocess_path, "r") as f:
        preprocess = json.load(f)

    return model, preprocess


def calculate_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    mdae = median_absolute_error(actual, predicted)
    return mae, mdae


def predict(model, embeddings, preprocess):
    # Standardize embeddings using training-set statistics
    x_mean = np.asarray(preprocess["x_mean"])
    x_scale = np.asarray(preprocess["x_scale"])

    X_scaled = ((embeddings - x_mean) / x_scale).astype(np.float32)

    X = torch.tensor(
        X_scaled,
        dtype=torch.float32,
        device=DEVICE
    )

    model.eval()

    with torch.no_grad():
        predictions = model(X).cpu().numpy()

    # Reverse target standardization
    predictions = predictions * preprocess["y_sd"] + preprocess["y_mu"]

    # Reverse log1p transformation
    if preprocess["log_target"]:
        predictions = np.expm1(predictions)

    return np.clip(predictions, 0.0, None)

# load project pairs

pairs_df = pd.read_csv(PAIRS_FILE)

required_pair_columns = {
    "pair",
    "source_project_id",
    "target_project_id"
}

if not required_pair_columns.issubset(pairs_df.columns):
    raise ValueError(
        f"Missing columns in project_pairs.csv: "
        f"{required_pair_columns - set(pairs_df.columns)}"
    )

if len(pairs_df) != 39:
    raise ValueError(
        f"Expected 39 project pairs, found {len(pairs_df)}"
    )



print(f"\nLoaded {len(pairs_df)} project pairs.")

# evaluate each pair

all_results = []

for _, pair_row in pairs_df.iterrows():
    pair = pair_row["pair"]
    source_id = str(pair_row["source_project_id"])
    target_id = str(pair_row["target_project_id"])

    print("\n" + "=" * 65)
    print(f"{pair}: {source_id} -> {target_id}")
    print("=" * 65)


    model_path = MODELS_DIR / pair / "story_point_cnn.pt"

    pair_split_dir = SPLITS_DIR / pair / "embeddings"

    target_embeddings_path = pair_split_dir / "target_test_embeddings.npy"
    target_metadata_path = pair_split_dir / "target_test_metadata.csv"

    # load target test embeddings and metadata

    X_test = np.load(target_embeddings_path)
    target_metadata = pd.read_csv(target_metadata_path)

    y_test = pd.to_numeric(
        target_metadata["story_points"],
        errors="coerce"
    ).to_numpy(dtype=np.float64)

    # validate
    if X_test.ndim != 2 or X_test.shape[1] != EMBEDDING_DIM:
        raise ValueError(
            f"{pair}: Unexpected target embedding shape "
            f"{X_test.shape}"
        )

    if len(X_test) != len(y_test):
        raise ValueError(
            f"{pair}: Embedding and label counts do not match."
        )

    if len(y_test) == 0:
        raise ValueError(f"{pair}: Target test set is empty.")

    if not np.isfinite(X_test).all():
        raise ValueError(
            f"{pair}: Target embeddings contain NaN or infinity."
        )

    if not np.isfinite(y_test).all():
        raise ValueError(
            f"{pair}: Target labels contain NaN or infinity."
        )

    if "project_id" in target_metadata.columns:
        target_projects = (
            target_metadata["project_id"].astype(str).unique()
        )

        if len(target_projects) != 1 or target_projects[0] != target_id:
            raise ValueError(
                f"{pair}: Target metadata contains unexpected "
                f"project IDs: {target_projects}"
            )

    print(f"Target test records: {len(y_test)}")
    print(f"Target embedding shape: {X_test.shape}")

    # load CNN and predict

    model, preprocess = load_model(model_path)
    cnn_prediction = predict(model, X_test, preprocess)

    # calculate eval metrics
    cnn_mae, cnn_mdae = calculate_metrics(y_test, cnn_prediction)

    # save individual pred
    if "issue_id" in target_metadata.columns:
        issue_ids = target_metadata["issue_id"].to_numpy()
    else:
        issue_ids = np.arange(len(y_test))

    predictions_df = pd.DataFrame({
        "issue_id" : issue_ids, 
        "actual_story_point" : y_test, 
        "cnn_prediction" : cnn_prediction,
        "cnn_absolute_error" : np.abs(y_test - cnn_prediction)
    })
    
    predictions_path = OUTPUT_DIR / f"{pair}_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)

    # store pair level results

    result = {
        "pair": pair,
        "source_project_id": source_id,
        "target_project_id": target_id,
        "target_test_records": len(y_test),
        "cnn_mae": cnn_mae,
        "cnn_mdae": cnn_mdae,
    }

    all_results.append(result)

    # display results

    print("\nEvaluation results:")
    print(f"CNN MAE: {cnn_mae:.4f}")
    print(f"CNN MdAE: {cnn_mdae:.4f}")

    print(f"Predictions saved: {predictions_path}")

# save results
results_df = pd.DataFrame(all_results)
summary_path = OUTPUT_DIR / "cross_project_evaluation_summary.csv"
results_df.to_csv(summary_path, index=False)

# overall summary

print("\n" + "=" * 65)
print("CROSS-PROJECT EVALUATION COMPLETED")
print("=" * 65)

print(f"Pairs evaluated: {len(results_df)}")

print("\nMacro-average results across project pairs:")
print(
    f"CNN MAE: "
    f"{results_df['cnn_mae'].mean():.4f}"
)

print(
    f"CNN MdAE: "
    f"{results_df['cnn_mdae'].mean():.4f}"
)

print(f"\nConsolidated results: {summary_path}")
print(f"Individual predictions: {OUTPUT_DIR}")

print("\nALL 39 PAIRS EVALUATED SUCCESSFULLY.")