import os
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# --------------------------------------------------
# PATHS AND SETTINGS
# --------------------------------------------------

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "processed")
SPLITS_DIR = os.path.join(DATA_DIR, "splits")
CACHE_DIR = os.path.join(DATA_DIR, "sbert_cache")

INPUT_FILE = os.path.join(DATA_DIR, "issues_clean.csv")

EMBEDDING_FILE = os.path.join(CACHE_DIR, "all_embeddings.npy")
CHECKPOINT_FILE = os.path.join(CACHE_DIR, "checkpoint.json")
INDEX_FILE = os.path.join(CACHE_DIR, "issue_index.csv")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 64
CHUNK_SIZE = 2000

os.makedirs(CACHE_DIR, exist_ok=True)

# --------------------------------------------------
# LOAD DATA AND MODEL
# --------------------------------------------------

print("Loading cleaned dataset...")

df = pd.read_csv(INPUT_FILE)

required = {"issue_id", "text", "story_points"}
if not required.issubset(df.columns):
    raise ValueError(f"Missing columns: {required - set(df.columns)}")

if df["issue_id"].astype(str).duplicated().any():
    raise ValueError("Duplicate issue_id values found.")

texts = df["text"].fillna("").astype(str).tolist()
issue_ids = df["issue_id"].astype(str).tolist()

print("Total issues:", len(df))

print("Loading SBERT model...")
model = SentenceTransformer(MODEL_NAME, device="cpu")

# --------------------------------------------------
# STEP A: ENCODE EACH ISSUE ONLY ONCE
# Supports resuming after interruption.
# --------------------------------------------------

total = len(df)
dimension = 384

if os.path.exists(EMBEDDING_FILE) and os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE, "r") as f:
        checkpoint = json.load(f)

    if checkpoint.get("total") == total and checkpoint.get("complete"):
        print("Global embeddings already completed.")
        start = total
    elif checkpoint.get("total") == total:
        start = checkpoint["completed_rows"]
        print("Resuming global encoding from row:", start)
    else:
        start = 0
else:
    start = 0

if start == 0:
    embeddings = np.lib.format.open_memmap(
        EMBEDDING_FILE,
        mode="w+",
        dtype="float32",
        shape=(total, dimension)
    )
else:
    embeddings = np.lib.format.open_memmap(
        EMBEDDING_FILE,
        mode="r+",
        dtype="float32",
        shape=(total, dimension)
    )

for begin in range(start, total, CHUNK_SIZE):
    end = min(begin + CHUNK_SIZE, total)

    print(f"\nEncoding issues {begin + 1} to {end} of {total}")

    vectors = model.encode(
        texts[begin:end],
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False
    )

    embeddings[begin:end] = vectors.astype("float32")
    embeddings.flush()

    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({
            "total": total,
            "completed_rows": end,
            "complete": end == total
        }, f)

    print("Saved checkpoint:", end)

del embeddings

# Save mapping between issue IDs and embedding row numbers
index_df = pd.DataFrame({
    "issue_id": issue_ids,
    "row_index": np.arange(total)
})
index_df.to_csv(INDEX_FILE, index=False)

print("\nGlobal SBERT encoding completed.")

# --------------------------------------------------
# STEP B: REUSE EMBEDDINGS FOR ALL 39 PAIRS
# --------------------------------------------------

all_embeddings = np.load(EMBEDDING_FILE, mmap_mode="r")

issue_to_index = dict(zip(
    index_df["issue_id"],
    index_df["row_index"]
))

pair_folders = sorted(
    [
        folder for folder in os.listdir(SPLITS_DIR)
        if folder.startswith("pair_")
        and os.path.isdir(os.path.join(SPLITS_DIR, folder))
    ]
)

print("\nTotal pairs found:", len(pair_folders))

for pair in pair_folders:
    pair_dir = os.path.join(SPLITS_DIR, pair)

    print("\nProcessing:", pair)

    for split_name in ["development", "target_test"]:

        csv_path = os.path.join(pair_dir, split_name + ".csv")

        if not os.path.exists(csv_path):
            raise FileNotFoundError(csv_path)

        pair_df = pd.read_csv(csv_path)

        if "issue_id" not in pair_df.columns:
            raise ValueError(f"issue_id missing in {csv_path}")

        ids = pair_df["issue_id"].astype(str).tolist()

        missing = [x for x in ids if x not in issue_to_index]

        if missing:
            raise ValueError(
                f"{len(missing)} issue IDs missing from global cache in {pair}"
            )

        indices = [issue_to_index[x] for x in ids]

        pair_vectors = np.asarray(
            all_embeddings[indices],
            dtype="float32"
        )

        output_dir = os.path.join(pair_dir, "embeddings")
        os.makedirs(output_dir, exist_ok=True)

        np.save(
            os.path.join(
                output_dir,
                split_name + "_embeddings.npy"
            ),
            pair_vectors
        )

        pair_df.to_csv(
            os.path.join(
                output_dir,
                split_name + "_metadata.csv"
            ),
            index=False
        )

        print(
            split_name,
            "shape:",
            pair_vectors.shape
        )

print("\nALL 39 PAIRS PROCESSED SUCCESSFULLY.")
