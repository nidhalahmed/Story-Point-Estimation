import os
import json
import shutil

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import mean_absolute_error, median_absolute_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# Configuration
# ============================================================
SEED = 42
N_PAIRS = 39
N_SPLITS = 5
INPUT_DIM = 384

EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.2
GRAD_CLIP = 1.0
SCHEDULER = dict(factor=0.5, patience=5, min_lr=1e-6)   # ReduceLROnPlateau

PATIENCE = 15
MIN_DELTA = 1e-4
MIN_FINAL_EPOCHS = 10   # floor on final-model epochs

LOG_TARGET = True       # train on standardized log1p(story_points)
LOSS = "huber"          # "huber" (standardized target) or "l1"

ROOT = os.getcwd()
SPLITS_DIR = os.path.join(ROOT, "data", "processed", "splits")
PAIRS_FILE = os.path.join(ROOT, "config", "project_pairs.csv")
MODELS_DIR = os.path.join(ROOT, "models")
MODEL_FILE = "story_point_cnn.pt"
PREPROCESS_FILE = "preprocess.json"

os.makedirs(MODELS_DIR, exist_ok=True)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Target transform and its inverse (identity when LOG_TARGET is False)
to_target, from_target = (np.log1p, np.expm1) if LOG_TARGET else (lambda a: a, lambda a: a)

pairs_df = pd.read_csv(PAIRS_FILE)
if len(pairs_df) != N_PAIRS:
    raise ValueError(f"Expected {N_PAIRS} project pairs, found {len(pairs_df)}")


# ============================================================
# Model
# ============================================================

class StoryPointCNN(nn.Module):
    """Three-layer 1D CNN for SBERT embeddings."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 16, 3, padding=1), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 8, 3, padding=1), nn.BatchNorm1d(8), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.regressor = nn.Sequential(
            nn.Linear(8 * (INPUT_DIM // 8), 32), nn.ReLU(), nn.Dropout(DROPOUT), nn.Linear(32, 1),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.regressor(self.features(x).flatten(1)).squeeze(-1)


# ============================================================
# Training and prediction
# ============================================================

def make_loader(X, y=None, shuffle=False):
    tensors = [torch.tensor(X, dtype=torch.float32)]
    if y is not None:
        tensors.append(torch.tensor(y, dtype=torch.float32))
    return DataLoader(
        TensorDataset(*tensors),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        pin_memory=(device.type == "cuda"),
    )


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        optimizer.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


def predict_points(model, prep, X):
    """Raw embeddings -> predicted story points (original scale)."""
    scaler, y_mu, y_sd = prep
    model.eval()
    loader = make_loader(scaler.transform(X).astype(np.float32))
    with torch.no_grad():
        z = torch.cat([model(xb.to(device)).cpu() for (xb,) in loader]).numpy()
    return np.clip(from_target(z.astype(np.float64) * y_sd + y_mu), 0.0, None)


def train_model(X_train, y_train, val=None, max_epochs=EPOCHS):
    """
    Train the CNN; returns (model, prep, best_epoch).

    val=(X_val, y_val): early stopping, LR scheduling and best-checkpoint
        restoring on validation MAE (original story-point units).
    val=None: train for exactly max_epochs (final model on all development data).
    prep=(scaler, y_mu, y_sd) is fit on X_train / y_train only.
    """
    torch.manual_seed(SEED)

    scaler = StandardScaler().fit(X_train)
    y_t = to_target(np.asarray(y_train, dtype=np.float64))
    y_mu, y_sd = float(y_t.mean()), float(y_t.std() + 1e-8)
    prep = (scaler, y_mu, y_sd)

    train_loader = make_loader(
        scaler.transform(X_train).astype(np.float32),
        ((y_t - y_mu) / y_sd).astype(np.float32),
        shuffle=True,
    )

    model = StoryPointCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.HuberLoss(delta=1.0) if LOSS == "huber" else nn.L1Loss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", **SCHEDULER)

    best_mae, best_epoch, stale, best_weights = float("inf"), 0, 0, None

    for epoch in range(1, max_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        if val is None:
            continue

        X_val, y_val = val
        val_mae = mean_absolute_error(y_val, predict_points(model, prep, X_val))
        scheduler.step(val_mae)

        if val_mae < best_mae - MIN_DELTA:
            best_mae, best_epoch, stale = val_mae, epoch, 0
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1

        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:03d}/{max_epochs} | Train Loss: {train_loss:.4f} | Val MAE: {val_mae:.4f}")

        if stale >= PATIENCE:
            print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch} | Best val MAE: {best_mae:.4f}")
            break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    return model, prep, best_epoch


# ============================================================
# Cross-validation and final training, per project pair
# ============================================================

def r4(values):
    return [round(float(v), 4) for v in values]


cv_summary = []

for _, row in pairs_df.iterrows():
    pair, source_id, target_id = row["pair"], row["source_project_id"], row["target_project_id"]
    print(f"\nProcessing {pair}: {source_id} -> {target_id}")

    emb_dir = os.path.join(SPLITS_DIR, pair, "embeddings")
    X_dev = np.load(os.path.join(emb_dir, "development_embeddings.npy"))
    dev_df = pd.read_csv(os.path.join(emb_dir, "development_metadata.csv"))

    if not (dev_df["project_id"] == source_id).all():
        raise ValueError(f"{pair}: Incorrect development project.")

    y_dev = dev_df["story_points"].to_numpy(dtype=np.float32)

    if len(X_dev) != len(y_dev):
        raise ValueError(f"{pair}: Embedding-label mismatch.")
    if len(y_dev) < N_SPLITS:
        raise ValueError(f"{pair}: Insufficient records for CV.")
    if X_dev.ndim != 2 or X_dev.shape[1] != INPUT_DIM:
        raise ValueError(f"{pair}: Expected embeddings with shape (N, {INPUT_DIM}), found {X_dev.shape}.")

    # 5-fold CV on the source project only
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    fold_mae, fold_mdae, fold_epochs, fold_stds = [], [], [], []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_dev), 1):
        X_val, y_val = X_dev[val_idx], y_dev[val_idx]
        model, prep, best_epoch = train_model(X_dev[train_idx], y_dev[train_idx], val=(X_val, y_val))

        preds = predict_points(model, prep, X_val)   # best checkpoint already restored
        fold_mae.append(mean_absolute_error(y_val, preds))
        fold_mdae.append(median_absolute_error(y_val, preds))
        fold_epochs.append(int(best_epoch))
        fold_stds.append(float(np.std(preds)))

        print(
            f"Fold {fold}/{N_SPLITS} | MAE: {fold_mae[-1]:.4f} | MdAE: {fold_mdae[-1]:.4f} | "
            f"Actual Std: {np.std(y_val):.4f} | Prediction Std: {fold_stds[-1]:.4f} | "
            f"Best Epoch: {best_epoch}"
        )

    mae_mean, mae_std = round(float(np.mean(fold_mae)), 4), round(float(np.std(fold_mae)), 4)
    mdae_mean, mdae_std = round(float(np.mean(fold_mdae)), 4), round(float(np.std(fold_mdae)), 4)
    std_mean = round(float(np.mean(fold_stds)), 4)
    final_epochs = min(EPOCHS, max(MIN_FINAL_EPOCHS, int(np.median(fold_epochs))))

    print(
        f"{pair} CV | MAE: {mae_mean:.4f} (+/- {mae_std:.4f}) | MdAE: {mdae_mean:.4f} (+/- {mdae_std:.4f}) | "
        f"Best epochs: {fold_epochs} | Final model: {final_epochs} epochs"
    )

    # Final model on all development data
    final_model, (scaler, y_mu, y_sd), _ = train_model(X_dev, y_dev, max_epochs=final_epochs)

    out_dir = os.path.join(MODELS_DIR, pair)
    os.makedirs(out_dir, exist_ok=True)

    torch.save(final_model.state_dict(), os.path.join(out_dir, MODEL_FILE))

    with open(os.path.join(out_dir, PREPROCESS_FILE), "w") as f:
        json.dump({
            "log_target": LOG_TARGET,
            "x_mean": scaler.mean_.tolist(),
            "x_scale": scaler.scale_.tolist(),
            "y_mu": y_mu,
            "y_sd": y_sd,
            "inverse": (
                "points = clip(expm1(z * y_sd + y_mu), 0)" if LOG_TARGET
                else "points = clip(z * y_sd + y_mu, 0)"
            ),
            "forward_x": "x_scaled = (x - x_mean) / x_scale",
        }, f)

    config = {
        "project": "Project 7",
        "setting": "SBERT + Cross-Project + Random Split",
        "pair": pair,
        "source_project_id": source_id,
        "target_project_id": target_id,
        "development_records": len(y_dev),
        "model_architecture": "StoryPointCNN (3-Layer 1D CNN, BatchNorm)",
        "input_dimension": INPUT_DIM,
        "max_epochs": EPOCHS,
        "final_training_epochs": final_epochs,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "dropout": DROPOUT,
        "loss": LOSS,
        "log_target": LOG_TARGET,
        "feature_standardization": "StandardScaler fit on training data",
        "early_stopping_patience": PATIENCE,
        "minimum_improvement": MIN_DELTA,
        "min_final_epochs": MIN_FINAL_EPOCHS,
        "learning_rate_scheduler": {"name": "ReduceLROnPlateau", **SCHEDULER},
        "gradient_clipping": GRAD_CLIP,
        "random_seed": SEED,
        "cv_validation_results": {
            "fold_mae": r4(fold_mae),
            "mean_mae": mae_mean,
            "std_mae": mae_std,
            "fold_mdae": r4(fold_mdae),
            "mean_mdae": mdae_mean,
            "std_mdae": mdae_std,
            "fold_best_epochs": fold_epochs,
            "fold_prediction_stds": r4(fold_stds),
            "mean_prediction_std": std_mean,
            "final_training_epochs": final_epochs,
        },
        "saved_model_file": MODEL_FILE,
        "preprocess_file": PREPROCESS_FILE,
    }
    with open(os.path.join(out_dir, "model_config.json"), "w") as f:
        json.dump(config, f, indent=4)

    cv_summary.append({
        "pair": pair,
        "source_project_id": source_id,
        "target_project_id": target_id,
        "development_records": len(y_dev),
        "cv_mean_mae": mae_mean,
        "cv_std_mae": mae_std,
        "cv_mean_mdae": mdae_mean,
        "cv_std_mdae": mdae_std,
        "mean_prediction_std": std_mean,
    })


# ============================================================
# Summary and handoff archive
# ============================================================

summary_df = pd.DataFrame(cv_summary)
summary_df.to_csv(os.path.join(MODELS_DIR, f"cv_summary_{N_PAIRS}_pairs.csv"), index=False)

print(f"\nTraining completed for all {N_PAIRS} project pairs.")
for label, col in [("MAE", "cv_mean_mae"), ("MdAE", "cv_mean_mdae")]:
    print(f"Overall CV {label}: {summary_df[col].mean():.4f} (std across pairs: {summary_df[col].std():.4f})")
print(f"Mean prediction std: {summary_df['mean_prediction_std'].mean():.4f}")

zip_file = shutil.make_archive(os.path.join(ROOT, "models_for_member_5"), "zip", MODELS_DIR)
print(f"\nHandoff archive created: {zip_file}")
print(f"Archive size: {os.path.getsize(zip_file) / (1024 ** 2):.2f} MB")