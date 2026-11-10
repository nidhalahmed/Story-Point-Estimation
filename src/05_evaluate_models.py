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
    """
    3-Layer 1D CNN regressor for SBERT sentence embeddings (384 dimensions).
    Conforms directly to Project Guidelines Section 3.1.
    """
    def __init__(self, input_dim=384, dropout=0.2):
        super(StoryPointCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)  # 384 -> 192

        self.conv2 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(2)  # 192 -> 96

        self.conv3 = nn.Conv1d(32, 16, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(2)  # 96 -> 48

        self.fc = nn.Sequential(
            nn.Linear(16 * (input_dim // 8), 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x).squeeze(-1)

# helper functions, load, metric_calc and predict


"""
Not sure about this. 
"""
def load_model(model_path):
    model = StoryPointCNN().to(DEVICE)

    try:
        checkpoint = torch.load(
            model_path,
            map_location=DEVICE,
            weights_only=True
        )
    except TypeError:
        checkpoint = torch.load(
            model_path,
            map_location=DEVICE
        )

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    return model

def calculate_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    mdae = median_absolute_error(actual, predicted)
    return mae, mdae

def predict(model, embeddings):
    X = torch.tensor(
        embeddings, dtype=torch.float32, device=DEVICE
    )

    model.eval()

    with torch.no_grad():
        predictions = model(X).cpu().numpy()

    return predictions

# load project pairs

pairs_df =pd.read_csv(PAIRS_FILE)

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


    model_path = (MODELS_DIR / pair / "story_point_cnn.pt")

    pair_split_dir = SPLITS_DIR / pair / "embeddings"

    target_embeddings_path = (pair_split_dir / "target_test_embeddings.npy")
    target_metadata_path = (pair_split_dir / "target_test_metadata.csv")

    # required_files = [
    #     model_path,
    #     config_path,
    #     target_embeddings_path,
    #     target_metadata_path
    # ]

    # missing_files = [
    #     str(path) for path in required_files
    #     if not path.exists()
    # ]

    # if missing_files:
    #     raise FileNotFoundError(
    #         f"Missing files for {pair}:\n"
    #         + "\n".join(missing_files)
    #     )

    # if "story_points" not in source_metadata.columns:
    #     raise ValueError(
    #         f"{pair}: story_points column missing "
    #         "from source development metadata."
    #     ) 

    # load target test embeddings and metadata

    X_test = np.load(target_embeddings_path)
    target_metadata = pd.read_csv(target_metadata_path)

    # if "story_points" not in target_metadata.columns:
    #     raise ValueError(
    #         f"{pair}: story_points column missing "
    #         "from target metadata."
    #     )

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

    model = load_model(model_path)
    cnn_prediction = predict(model, X_test)

    if len(cnn_prediction) != len(y_test):
        raise ValueError(
            f"{pair}: Prediction and target counts do not match."
        )

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
    
    predictions_path = (OUTPUT_DIR / f"{pair}_predictions.csv")
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

    # release model 
    del model

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

# save results
results_df = pd.DataFrame(all_results)
summary_path = (OUTPUT_DIR / "cross_project_evaluation_summary.csv")
results_df.to_csv(
    summary_path,
    index=False
)

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

