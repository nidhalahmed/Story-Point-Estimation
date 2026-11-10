import os
import json
import shutil

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# Configuration
# ============================================================
SEED = 42

# Training
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Early stopping
PATIENCE = 15
MIN_DELTA = 1e-4

# Architecture
INPUT_DIM = 384
DROPOUT = 0.2

# Cross-validation
N_SPLITS = 5

ROOT = os.getcwd()
DATA_DIR = os.path.join(ROOT, "data", "processed")
SPLITS_DIR = os.path.join(DATA_DIR, "splits")
PAIRS_FILE = os.path.join(ROOT, "config", "project_pairs.csv")
MODELS_DIR = os.path.join(ROOT, "models")

os.makedirs(MODELS_DIR, exist_ok=True)

# Reproducibility
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ============================================================
# Load Project Pairs
# ============================================================

pairs_df = pd.read_csv(PAIRS_FILE)

if len(pairs_df) != 39:
    raise ValueError(f"Expected 39 project pairs, found {len(pairs_df)}")


# ============================================================
# CNN Architecture
# ============================================================

class StoryPointCNN(nn.Module):
    """Three-layer 1D CNN for SBERT embeddings."""

    def __init__(self, input_dim=INPUT_DIM, dropout=0.2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.regressor = nn.Sequential(
            nn.Linear(16 * (input_dim // 8), 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.features(x)
        x = x.flatten(start_dim=1)

        return self.regressor(x).squeeze(-1)

# ============================================================
# Training and Prediction Functions
# ============================================================

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)

        loss.backward()

        # Prevent unusually large gradients.
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        total_loss += loss.item() * len(y_batch)

    return total_loss / len(loader.dataset)


def predict(model, loader):
    model.eval()
    predictions = []

    with torch.no_grad():
        for (X_batch,) in loader:
            X_batch = X_batch.to(device)

            batch_predictions = model(X_batch)

            predictions.extend(
                batch_predictions.cpu().numpy()
            )

    return np.array(predictions)


def create_loader(X, y=None, shuffle=False):

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32
    ).unsqueeze(1)

    if y is not None:
        y_tensor = torch.tensor(
            y,
            dtype=torch.float32
        )

        dataset = TensorDataset(X_tensor, y_tensor)

    else:
        dataset = TensorDataset(X_tensor)

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        pin_memory=(device.type == "cuda")
    )


def train_model(
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    max_epochs=EPOCHS,
    patience=PATIENCE
):
    """
    Train the CNN with optional validation-based early stopping.

    When validation data is supplied:
      - Monitor validation MAE.
      - Save the best model weights.
      - Reduce learning rate when validation stops improving.
      - Stop after PATIENCE unsuccessful epochs.

    When validation data is absent:
      - Train for the specified fixed number of epochs.
      - Used for final model training.
    """

    train_loader = create_loader(
        X_train,
        y_train,
        shuffle=True
    )

    model = StoryPointCNN(
        dropout=DROPOUT
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    criterion = nn.L1Loss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )

    val_loader = None

    if X_val is not None and y_val is not None:
        val_loader = create_loader(X_val)

    best_val_mae = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    best_weights = None

    history = {
        "train_loss": [],
        "val_mae": [],
        "learning_rate": []
    }

    for epoch in range(1, max_epochs + 1):

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion
        )

        history["train_loss"].append(train_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        history["learning_rate"].append(current_lr)

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        if val_loader is not None:

            val_predictions = predict(
                model,
                val_loader
            )

            val_mae = mean_absolute_error(
                y_val,
                val_predictions
            )

            history["val_mae"].append(float(val_mae))

            scheduler.step(val_mae)

            if val_mae < best_val_mae - MIN_DELTA:

                best_val_mae = val_mae
                best_epoch = epoch

                best_weights = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }

                epochs_without_improvement = 0

            else:
                epochs_without_improvement += 1

            if epoch == 1 or epoch % 10 == 0:

                print(
                    f"Epoch {epoch:03d}/{max_epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val MAE: {val_mae:.4f} | "
                    f"LR: {current_lr:.6f}"
                )

            if epochs_without_improvement >= patience:

                print(
                    f"Early stopping at epoch {epoch}. "
                    f"Best epoch: {best_epoch} | "
                    f"Best validation MAE: {best_val_mae:.4f}"
                )

                break

        else:

            if epoch == 1 or epoch % 10 == 0:

                print(
                    f"Epoch {epoch:03d}/{max_epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"LR: {current_lr:.6f}"
                )

    # --------------------------------------------------------
    # Restore best validation checkpoint
    # --------------------------------------------------------

    if best_weights is not None:

        model.load_state_dict(best_weights)

    if val_loader is not None:

        print(
            f"Training finished | "
            f"Best epoch: {best_epoch} | "
            f"Best validation MAE: {best_val_mae:.4f}"
        )

    else:
        best_epoch = max_epochs

    model.training_history = history
    model.best_epoch = best_epoch
    model.best_val_mae = (
        best_val_mae if val_loader is not None else None
    )

    return model



# ============================================================
# Cross-Validation and Final Training
# ============================================================

cv_summary = []

for _, pair_row in pairs_df.iterrows():

    pair_name = pair_row["pair"]
    source_id = pair_row["source_project_id"]
    target_id = pair_row["target_project_id"]

    print(f"\nProcessing {pair_name}: {source_id} -> {target_id}")

    embedding_dir = os.path.join(
        SPLITS_DIR, pair_name, "embeddings"
    )

    # Load development data
    X_dev = np.load(
        os.path.join(embedding_dir, "development_embeddings.npy")
    )

    dev_df = pd.read_csv(
        os.path.join(embedding_dir, "development_metadata.csv")
    )

    if not (dev_df["project_id"] == source_id).all():
        raise ValueError(f"{pair_name}: Incorrect development project.")

    y_dev = dev_df["story_points"].to_numpy(dtype=np.float32)

    if len(X_dev) != len(y_dev):
        raise ValueError(f"{pair_name}: Embedding-label mismatch.")

    if len(y_dev) < N_SPLITS:
        raise ValueError(f"{pair_name}: Insufficient records for CV.")

    if X_dev.ndim != 2 or X_dev.shape[1] != INPUT_DIM:
        raise ValueError(
            f"{pair_name}: Expected embeddings with shape (N, {INPUT_DIM}), "
            f"found {X_dev.shape}."
        )

    # --------------------------------------------------------
    # 5-Fold Cross-Validation
    # --------------------------------------------------------

    kf = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=SEED
    )


    fold_mae = []
    fold_best_epochs = []
    fold_prediction_stds = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_dev), 1):

        print(f"\n--- Fold {fold}/{N_SPLITS} ---")

        X_train, X_val = X_dev[train_idx], X_dev[val_idx]
        y_train, y_val = y_dev[train_idx], y_dev[val_idx]

        model = train_model(
            X_train,
            y_train,
            X_val,
            y_val,
            max_epochs=EPOCHS,
            patience=PATIENCE
        )

        # Best validation checkpoint is already restored.
        val_loader = create_loader(X_val)

        val_predictions = predict(model, val_loader)

        mae = mean_absolute_error(y_val, val_predictions)

        prediction_std = float(np.std(val_predictions))
        actual_std = float(np.std(y_val))

        fold_mae.append(mae)
        fold_best_epochs.append(model.best_epoch)
        fold_prediction_stds.append(prediction_std)

        print(
            f"Fold {fold}/{N_SPLITS} | "
            f"MAE: {mae:.4f} | "
            f"Actual Std: {actual_std:.4f} | "
            f"Prediction Std: {prediction_std:.4f} | "
            f"Best Epoch: {model.best_epoch}"
        )

        del model

    cv_mean_mae = float(np.mean(fold_mae))
    cv_std_mae = float(np.std(fold_mae))

    # Median best epoch across all folds.
    final_epochs = max(
        1,
        int(np.median(fold_best_epochs))
    )

    print(f"\nCV Mean MAE: {cv_mean_mae:.4f}")
    print(f"CV Std MAE: {cv_std_mae:.4f}")
    print(f"Best epochs per fold: {fold_best_epochs}")
    print(f"Final training epochs: {final_epochs}")


    cv_mean_mae = float(np.mean(fold_mae))
    cv_std_mae = float(np.std(fold_mae))

    # --------------------------------------------------------
    # Train Final Model on All Development Data
    # --------------------------------------------------------

    print(
        f"\nTraining final model for {final_epochs} epochs "
        f"on all development data."
    )

    final_model = train_model(
        X_dev,
        y_dev,
        max_epochs=final_epochs
    )
    # --------------------------------------------------------
    # Save Model and Configuration
    # --------------------------------------------------------

    pair_model_dir = os.path.join(MODELS_DIR, pair_name)
    os.makedirs(pair_model_dir, exist_ok=True)

    model_filename = "story_point_cnn.pt"

    torch.save(
        final_model.state_dict(),
        os.path.join(pair_model_dir, model_filename)
    )

    config = {
        "project": "Project 7",
        "setting": "SBERT + Cross-Project + Random Split",
        "pair": pair_name,
        "source_project_id": source_id,
        "target_project_id": target_id,
        "development_records": len(y_dev),
        "model_architecture": "StoryPointCNN (3-Layer 1D CNN)",
        "input_dimension": INPUT_DIM,
        "max_epochs": EPOCHS,
        "final_training_epochs": final_epochs,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "dropout": DROPOUT,
        "early_stopping_patience": PATIENCE,
        "minimum_improvement": MIN_DELTA,
        "learning_rate_scheduler": "ReduceLROnPlateau",
        "gradient_clipping": 1.0,
        "fold_best_epochs": fold_best_epochs,
        "random_seed": SEED,
"cv_validation_results": {
    "fold_mae": [round(float(x), 4) for x in fold_mae],
    "mean_mae": round(cv_mean_mae, 4),
    "std_mae": round(cv_std_mae, 4),
    "fold_best_epochs": fold_best_epochs,
    "fold_prediction_stds": [
        round(float(x), 4)
        for x in fold_prediction_stds
    ],
    "final_training_epochs": final_epochs
},
        "saved_model_file": model_filename
    }

    with open(
        os.path.join(pair_model_dir, "model_config.json"), "w"
    ) as f:
        json.dump(config, f, indent=4)

    cv_summary.append({
        "pair": pair_name,
        "source_project_id": source_id,
        "target_project_id": target_id,
        "development_records": len(y_dev),
        "cv_mean_mae": round(cv_mean_mae, 4),
        "cv_std_mae": round(cv_std_mae, 4)
    })

    print(
        f"{pair_name} completed - "
        f"Mean CV MAE: {cv_mean_mae:.4f} "
        f"(+/- {cv_std_mae:.4f})"
    )

    del final_model

# ============================================================
# Save Cross-Validation Summary
# ============================================================

summary_df = pd.DataFrame(cv_summary)

summary_df.to_csv(
    os.path.join(MODELS_DIR, "cv_summary_39_pairs.csv"),
    index=False
)

print("\nTraining completed for all 39 project pairs.")
print(f"Overall CV MAE: {summary_df['cv_mean_mae'].mean():.4f}")
print(f"Overall CV MAE Std: {summary_df['cv_mean_mae'].std():.4f}")


# ============================================================
# Create Handoff ZIP for Member 5
# ============================================================

zip_path = os.path.join(ROOT, "models_for_member_5")

shutil.make_archive(zip_path, "zip", MODELS_DIR)

zip_file = zip_path + ".zip"

print(f"\nHandoff archive created: {zip_file}")
print(f"Archive size: {os.path.getsize(zip_file) / (1024 ** 2):.2f} MB")