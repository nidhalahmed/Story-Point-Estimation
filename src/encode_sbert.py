
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ==========================================
# 1. CONFIGURATION
# ==========================================

SPLITS_DIR = (
    r"C:\Users\DELL\Downloads\sbert-story-point-estimation"
    r"\sbert-story-point-estimation\data\processed\splits"
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 64

# ==========================================
# 2. LOAD SBERT MODEL
# ==========================================

print("Loading SBERT model...")

model = SentenceTransformer(MODEL_NAME)

print("Model loaded!")
print("Embedding dimension:",
      model.get_sentence_embedding_dimension())


# ==========================================
# 3. ENCODE ONE CSV FILE
# ==========================================

def encode_file(csv_path, output_dir, file_type):

    embedding_path = os.path.join(
        output_dir, file_type + "_embeddings.npy"
    )

    metadata_path = os.path.join(
        output_dir, file_type + "_metadata.csv"
    )

    # Skip if both outputs already exist
    if os.path.exists(embedding_path) and os.path.exists(metadata_path):
        print(f"Already completed: {file_type}")
        return

    print(f"\nReading {file_type} data...")

    df = pd.read_csv(csv_path)

    if "text" not in df.columns:
        raise ValueError(f"'text' column missing in {csv_path}")

    if "story_points" not in df.columns:
        raise ValueError(f"'story_points' column missing in {csv_path}")

    # Preserve original row order
    df = df.reset_index(drop=True)

    # Prepare issue descriptions
    texts = df["text"].fillna("").astype(str).tolist()

    print("Number of descriptions:", len(texts))

    # Generate SBERT embeddings
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    # Save embeddings
    np.save(embedding_path, embeddings)

    # Save corresponding original records
    df.to_csv(metadata_path, index=False)

    print("Embedding shape:", embeddings.shape)
    print("Saved:", embedding_path)
    print("Saved:", metadata_path)


# ==========================================
# 4. FIND ALL 39 PAIRS
# ==========================================

pair_folders = sorted(
    folder for folder in os.listdir(SPLITS_DIR)
    if folder.startswith("pair_")
    and os.path.isdir(os.path.join(SPLITS_DIR, folder))
)

print("\nPair folders found:", len(pair_folders))

if len(pair_folders) != 39:
    raise ValueError(
        f"Expected 39 pairs, found {len(pair_folders)}"
    )


# ==========================================
# 5. PROCESS EACH PAIR
# ==========================================

for pair in tqdm(pair_folders, desc="Processing pairs"):

    print(f"\n========== {pair} ==========")

    pair_dir = os.path.join(SPLITS_DIR, pair)

    development_csv = os.path.join(
        pair_dir, "development.csv"
    )

    target_test_csv = os.path.join(
        pair_dir, "target_test.csv"
    )

    if not os.path.exists(development_csv):
        raise FileNotFoundError(development_csv)

    if not os.path.exists(target_test_csv):
        raise FileNotFoundError(target_test_csv)

    output_dir = os.path.join(pair_dir, "embeddings")
    os.makedirs(output_dir, exist_ok=True)

    # Encode development data
    encode_file(
        development_csv,
        output_dir,
        "development"
    )

    # Encode target test data
    encode_file(
        target_test_csv,
        output_dir,
        "target_test"
    )


# ==========================================
# 6. COMPLETION
# ==========================================

print("\n===================================")
print("SBERT ENCODING COMPLETED")
print("All 39 pairs processed.")
print("===================================")
