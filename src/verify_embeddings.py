import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(ROOT, "data", "processed")
SPLITS_DIR = os.path.join(DATA_DIR, "splits")
CACHE_DIR = os.path.join(DATA_DIR, "sbert_cache")

print("Checking global embedding cache...")

global_embeddings = np.load(
    os.path.join(CACHE_DIR, "all_embeddings.npy"),
    mmap_mode="r"
)

issue_index = pd.read_csv(
    os.path.join(CACHE_DIR, "issue_index.csv")
)

print("Global embedding shape:", global_embeddings.shape)
print("Issue index rows:", len(issue_index))

errors = []

if global_embeddings.shape != (len(issue_index), 384):
    errors.append("Global embedding dimensions do not match index.")

if issue_index["issue_id"].astype(str).duplicated().any():
    errors.append("Duplicate issue IDs in global index.")

folders = sorted(
    f for f in os.listdir(SPLITS_DIR)
    if f.startswith("pair_")
    and os.path.isdir(os.path.join(SPLITS_DIR, f))
)

print("\nPairs found:", len(folders))

if len(folders) != 39:
    errors.append(f"Expected 39 pairs, found {len(folders)}.")

for pair in folders:

    pair_dir = os.path.join(SPLITS_DIR, pair)
    embedding_dir = os.path.join(pair_dir, "embeddings")

    for split in ["development", "target_test"]:

        original = pd.read_csv(
            os.path.join(pair_dir, split + ".csv")
        )

        if original["project_id"].nunique() != 1:
            errors.append(f"{pair}: {split} contains multiple projects.")

        metadata = pd.read_csv(
            os.path.join(
                embedding_dir,
                split + "_metadata.csv"
            )
        )

        X = np.load(
            os.path.join(
                embedding_dir,
                split + "_embeddings.npy"
            ),
            mmap_mode="r"
        )

        if len(original) != len(metadata):
            errors.append(f"{pair}: {split} metadata row mismatch")

        if X.shape != (len(original), 384):
            errors.append(f"{pair}: {split} embedding shape mismatch")

        if not original["issue_id"].astype(str).equals(
            metadata["issue_id"].astype(str)
        ):
            errors.append(f"{pair}: {split} issue ID order mismatch")

        if not np.isfinite(X).all():
            errors.append(f"{pair}: {split} contains invalid values")

    print(pair, "verified")

print("\n--------------------------")

if errors:
    print("VERIFICATION FAILED:")
    for error in errors:
        print("-", error)
else:
    print("SUCCESS: All 39 pairs verified!")
    print("Embedding dimension: 384")
    print("Metadata alignment: checked")
    print("No invalid embedding values found.")