"""
02_prepare_data.py
==================
Member 2 — Cleaning and splitting

Purpose
-------
Produce a cleaned dataset and the approved Leave-One-Project-Out (LOPO)
development / test splits for the TAWOS story-point estimation study.

Experimental design
-------------------
39 projects → 39 pairs (pair_01 … pair_39, sorted alphabetically by
project_id).

For each pair:
  • target_test.csv  — all labelled issues from the held-out project.
  • development.csv  — all labelled issues from every other project.
  • split_metadata.json — reproducibility record.

The test set is NEVER touched during training or model selection.

Inputs
------
  data/processed/issues_raw.csv        (produced by 01_load_data.py)

Outputs
-------
  data/processed/issues_clean.csv
  data/processed/cleaning_report.json
  data/processed/project_summary_after_cleaning.csv
  data/processed/story_point_distribution.csv
  data/processed/potential_outliers.csv
  data/processed/duplicate_issue_ids.csv
  data/processed/splits/pair_NN/development.csv
  data/processed/splits/pair_NN/target_test.csv
  data/processed/splits/pair_NN/split_metadata.json

Usage
-----
  python src/02_prepare_data.py           # run from repository root
  python src/02_prepare_data.py --dry-run # inspect exclusions without saving
"""

# ============================================================
# 0. IMPORTS AND ARGUMENT PARSING
# ============================================================

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clean the raw TAWOS issue dataset and produce "
            "Leave-One-Project-Out cross-project splits."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print exclusion counts and split sizes but do not "
            "write any files to disk."
        ),
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help=(
            "Integer random seed used throughout this script. "
            "Applies to any shuffle operations (default: 42)."
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=(
            "Override the default input path "
            "(data/processed/issues_raw.csv)."
        ),
    )
    return parser.parse_args()


# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

# Project root is the parent of the directory that contains this script.
ROOT_DIR: Path = Path(__file__).resolve().parents[1]

REQUIRED_COLUMNS: list[str] = [
    "issue_id",
    "issue_key",
    "project_id",
    "project_name",
    "text",
    "story_points",
]


# ============================================================
# 2. TEXT NORMALISATION
# ============================================================

def clean_text(value: object) -> str:
    """
    Conservative text normalisation applied to the issue text field.

    Rules applied (in order):
    1. NaN / None → empty string.
    2. Normalise Windows (\\r\\n) and old Mac (\\r) line endings to \\n.
    3. Collapse runs of spaces and tabs to a single space.
    4. Collapse three-or-more consecutive blank lines to two blank lines
       (preserving paragraph structure).
    5. Strip leading / trailing whitespace.

    Intentionally NOT applied:
    - Lower-casing (SBERT is case-sensitive at the sub-word level).
    - Punctuation removal (punctuation carries meaning in issue text).
    - Stop-word removal (handled by the language model).
    """
    if pd.isna(value):
        return ""

    text = str(value)

    # Normalise line endings.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse horizontal whitespace.
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse excessive blank lines.
    text = re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", text)

    return text.strip()


# ============================================================
# 3. LOADING
# ============================================================

def load_raw(input_path: Path) -> pd.DataFrame:
    """Load the raw CSV produced by 01_load_data.py."""
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}\n"
            "Run src/01_load_data.py first."
        )

    print(f"Loading raw data from: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)
    print(f"  -> {len(df):,} records loaded.")
    return df


# ============================================================
# 4. SCHEMA VALIDATION
# ============================================================

def validate_schema(df: pd.DataFrame) -> None:
    """Raise ValueError if any required column is absent."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Required columns missing from input CSV: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )
    print("  Schema OK — all required columns present.")


# ============================================================
# 5. IDENTIFIER NORMALISATION AND DUPLICATE DETECTION
# ============================================================

def normalise_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    """Cast identifier columns to string and strip whitespace."""
    df = df.copy()
    df["issue_id"] = df["issue_id"].astype("string").str.strip()
    df["project_id"] = df["project_id"].astype("string").str.strip()
    return df


def detect_duplicates(
    df: pd.DataFrame,
    output_dir: Path,
    dry_run: bool,
) -> tuple[pd.DataFrame, int]:
    """
    Identify rows sharing the same issue_id.

    Policy
    ------
    Duplicate issue IDs indicate a data integrity problem in the
    upstream extract. The script saves the offending rows for
    inspection and raises an error so the pipeline stops cleanly
    rather than silently using ambiguous records.

    Returns
    -------
    duplicate_rows : DataFrame of all rows involved in a duplicate.
    duplicate_count : number of *unique* issue IDs that are duplicated.
    """
    missing_id_mask = df["issue_id"].isna() | df["issue_id"].eq("")

    dup_mask = ~missing_id_mask & df.duplicated(subset=["issue_id"], keep=False)
    dup_rows = df.loc[dup_mask].copy()
    dup_count = int(dup_rows["issue_id"].nunique())

    if not dry_run:
        dup_rows.to_csv(output_dir / "duplicate_issue_ids.csv", index=False)

    if dup_count > 0:
        raise ValueError(
            f"{dup_count} issue ID(s) appear more than once. "
            "Review data/processed/duplicate_issue_ids.csv before "
            "continuing. The upstream extract should deduplicate issues "
            "or the duplicate policy should be explicitly stated."
        )

    print(f"  Duplicate issue IDs: {dup_count}  (none found — OK).")
    return dup_rows, dup_count


# ============================================================
# 6. EXCLUSION RULES
# ============================================================

def compute_exclusion_masks(
    df: pd.DataFrame,
) -> dict[str, pd.Series]:
    """
    Compute one boolean mask per exclusion rule.

    Exclusion rules (documented)
    ----------------------------
    E1  missing_issue_id    : issue_id is null or empty string.
    E2  missing_project_id  : project_id is null or empty string.
    E3  missing_sp          : story_points was NULL in the database.
    E4  sp_parse_failure    : story_points was non-null but could not be
                              cast to a finite float (e.g. 'TBD').
    E5  non_finite_sp       : story_points parsed to +/-Inf.
    E6  negative_sp         : story_points < 0 (meaningless for effort).
    E7  zero_sp             : story_points == 0.  Retained as a *flag*
                              here but excluded from the final clean set;
                              zero often signals "not estimated" rather
                              than "zero effort".
    E8  empty_text          : text field is empty after normalisation.

    Notes
    -----
    Rules E3–E6 and E8 make a row unusable as a training or test
    example.  Rule E7 (zero story points) is treated as a separate,
    auditable exclusion because project conventions differ.
    """
    # --- Identifier checks ---
    e1 = df["issue_id"].isna() | df["issue_id"].eq("")
    e2 = df["project_id"].isna() | df["project_id"].eq("")

    # --- Story-point checks ---
    # Apply to the ORIGINAL column before numeric conversion.
    original_sp = df["story_points"].copy()
    numeric_sp = pd.to_numeric(original_sp, errors="coerce")

    e3 = original_sp.isna()
    e4 = original_sp.notna() & numeric_sp.isna()
    e5 = numeric_sp.notna() & ~np.isfinite(numeric_sp.where(numeric_sp.notna()))
    e6 = numeric_sp.notna() & numeric_sp.lt(0)
    e7 = numeric_sp.notna() & numeric_sp.eq(0)

    # --- Text check (after normalisation) ---
    e8 = df["text"].eq("")

    return {
        "missing_issue_id": e1,
        "missing_project_id": e2,
        "missing_story_points": e3,
        "story_point_parse_failure": e4,
        "non_finite_story_points": e5,
        "negative_story_points": e6,
        "zero_story_points": e7,
        "empty_text": e8,
    }


def apply_exclusions(
    df: pd.DataFrame,
    masks: dict[str, pd.Series],
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Apply all exclusion rules and return the cleaned DataFrame.

    Returns
    -------
    clean_df         : rows that pass all filters.
    exclude_any      : union mask of E1–E6, E8 (hard exclusions).
    zero_sp_mask     : mask for E7 (soft exclusion, reported separately).
    """
    hard_exclusion_keys = [
        "missing_issue_id",
        "missing_project_id",
        "missing_story_points",
        "story_point_parse_failure",
        "non_finite_story_points",
        "negative_story_points",
        "empty_text",
    ]

    exclude_hard = pd.Series(False, index=df.index)
    for key in hard_exclusion_keys:
        exclude_hard |= masks[key]

    zero_sp_mask = masks["zero_story_points"] & ~exclude_hard

    # Exclude both hard failures AND zero story points.
    exclude_any = exclude_hard | zero_sp_mask

    clean_df = df.loc[~exclude_any].copy()

    # Assign the numeric story_points column to the cleaned slice.
    clean_df["story_points"] = pd.to_numeric(
        clean_df["story_points"], errors="coerce"
    )

    return clean_df, exclude_any, zero_sp_mask


# ============================================================
# 7. OUTLIER FLAGGING
# ============================================================

def flag_outliers(
    clean_df: pd.DataFrame,
) -> tuple[pd.DataFrame, float | None]:
    """
    Flag issues whose story-point value is at or above the 99th
    percentile.  These rows are saved for review but NOT removed
    from the clean dataset.
    """
    if clean_df.empty:
        return clean_df.copy(), None

    p99 = float(clean_df["story_points"].quantile(0.99))
    outlier_df = (
        clean_df.loc[clean_df["story_points"] >= p99]
        .sort_values("story_points", ascending=False)
    )
    return outlier_df, p99

def remove_outliers(
    clean_df: pd.DataFrame,
    outlier_df: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """
    Remove all issues identified in potential_outliers.csv
    from the cleaned dataset using issue_id.

    Returns
    -------
    filtered_df : DataFrame without flagged outliers.
    removed_count : number of removed records.
    """

    outlier_ids = set(outlier_df["issue_id"].astype(str))

    filtered_df = clean_df[
        ~clean_df["issue_id"].astype(str).isin(outlier_ids)
    ].copy()

    removed_count = len(clean_df) - len(filtered_df)

    return filtered_df, removed_count


# ============================================================
# 8. SUMMARIES
# ============================================================

def project_summary(clean_df: pd.DataFrame) -> pd.DataFrame:
    """Return a per-project aggregation over the cleaned dataset."""
    return (
        clean_df
        .groupby(["project_id", "project_name"], sort=False)
        .agg(
            labelled_issues=("issue_id", "count"),
            unique_story_point_values=("story_points", "nunique"),
            minimum_story_points=("story_points", "min"),
            median_story_points=("story_points", "median"),
            maximum_story_points=("story_points", "max"),
        )
        .reset_index()
        .sort_values("labelled_issues", ascending=False)
        .reset_index(drop=True)
    )


def story_point_distribution(clean_df: pd.DataFrame) -> pd.DataFrame:
    """Return a frequency table of story-point values."""
    return (
        clean_df["story_points"]
        .value_counts()
        .sort_index()
        .rename_axis("story_points")
        .reset_index(name="issue_count")
    )


# ============================================================
# 9. SPECIFIED SOURCE-TARGET SPLITS
# ============================================================

def create_pair_splits(
    clean_df: pd.DataFrame,
    splits_dir: Path,
    pairs_file: Path,
    dry_run: bool,
) -> list[dict]:
    """
    Create the 39 explicitly assigned cross-project splits.

    For each pair:
      - development.csv: issues from the specified source project only.
      - target_test.csv: issues from the specified target project only.

    All other projects are excluded from that pair.
    """

    if not pairs_file.exists():
        raise FileNotFoundError(
            f"Pair configuration not found: {pairs_file}"
        )

    pairs_df = pd.read_csv(pairs_file, dtype=str)

    required = {
        "pair",
        "source_project_id",
        "target_project_id",
    }

    if not required.issubset(pairs_df.columns):
        raise ValueError(
            f"Pair configuration must contain: {sorted(required)}"
        )

    pairs_df = pairs_df[list(required)].copy()

    for col in required:
        pairs_df[col] = pairs_df[col].str.strip()

    if pairs_df.isna().any().any() or (pairs_df == "").any().any():
        raise ValueError("Pair configuration contains missing values.")

    if pairs_df["pair"].duplicated().any():
        raise ValueError("Duplicate pair IDs found.")

    if len(pairs_df) != 39:
        raise ValueError(
            f"Expected exactly 39 pairs, found {len(pairs_df)}."
        )

    available_projects = set(clean_df["project_id"].dropna().unique())

    missing_projects = (
        set(pairs_df["source_project_id"])
        | set(pairs_df["target_project_id"])
    ) - available_projects

    if missing_projects:
        raise ValueError(
            f"Projects missing from cleaned dataset: "
            f"{sorted(missing_projects)}"
        )

    if (
        pairs_df["source_project_id"]
        == pairs_df["target_project_id"]
    ).any():
        raise ValueError("A pair cannot have identical source and target.")

    print(f"\n  Creating {len(pairs_df)} specified pair splits...")

    split_records = []

    for _, pair in pairs_df.iterrows():

        pair_id = pair["pair"]
        source_id = pair["source_project_id"]
        target_id = pair["target_project_id"]

        development_df = clean_df[
            clean_df["project_id"] == source_id
        ].copy()

        target_test_df = clean_df[
            clean_df["project_id"] == target_id
        ].copy()

        if development_df.empty or target_test_df.empty:
            raise ValueError(
                f"{pair_id}: Empty source or target dataset."
            )

        # Verify project purity.
        assert development_df["project_id"].eq(source_id).all()
        assert target_test_df["project_id"].eq(target_id).all()

        assert not set(development_df["issue_id"]).intersection(
            set(target_test_df["issue_id"])
        )

        source_name = development_df["project_name"].iloc[0]
        target_name = target_test_df["project_name"].iloc[0]

        metadata = {
            "pair": pair_id,
            "source_project_id": source_id,
            "source_project_name": source_name,
            "target_project_id": target_id,
            "target_project_name": target_name,
            "development_records": int(len(development_df)),
            "target_test_records": int(len(target_test_df)),
            "note": (
                "Development contains only the specified source project. "
                "Target test contains only the specified target project. "
                "All other projects are excluded from this pair."
            ),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        split_records.append(metadata)

        if dry_run:
            print(
                f"    [DRY RUN] {pair_id}: "
                f"{source_id} ({len(development_df):,}) -> "
                f"{target_id} ({len(target_test_df):,})"
            )
            continue

        pair_dir = splits_dir / pair_id
        pair_dir.mkdir(parents=True, exist_ok=True)

        development_df.to_csv(
            pair_dir / "development.csv", index=False
        )

        target_test_df.to_csv(
            pair_dir / "target_test.csv", index=False
        )

        with open(
            pair_dir / "split_metadata.json",
            "w",
            encoding="utf-8",
        ) as fh:
            json.dump(metadata, fh, indent=4)

        print(
            f"    {pair_id}: "
            f"{source_id} ({len(development_df):,}) -> "
            f"{target_id} ({len(target_test_df):,})"
        )

    return split_records


# ============================================================
# 10. CLEANING REPORT
# ============================================================

def build_cleaning_report(
    initial_rows: int,
    df: pd.DataFrame,
    masks: dict[str, pd.Series],
    exclude_any: pd.Series,
    zero_sp_mask: pd.Series,
    clean_df: pd.DataFrame,
    duplicate_count: int,
    p99: float | None,
    split_records: list[dict],
    random_seed: int,
) -> dict:
    """Assemble the complete cleaning and split report."""
    return {
        "script": "02_prepare_data.py",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "random_seed": random_seed,

        # --- Row counts ---
        "input_records": int(initial_rows),
        "records_retained": int(len(clean_df)),
        "records_excluded_total": int(exclude_any.sum()),

        # --- Identifier audit ---
        "unique_issue_ids_in_raw": int(
            df["issue_id"].nunique(dropna=True)
        ),
        "projects_in_raw_data": int(
            df["project_id"].nunique(dropna=True)
        ),
        "projects_after_cleaning": int(
            clean_df["project_id"].nunique()
        ),
        "duplicate_issue_ids": int(duplicate_count),

        # --- Per-rule exclusion counts ---
        "exclusions": {
            "E1_missing_issue_id": int(masks["missing_issue_id"].sum()),
            "E2_missing_project_id": int(masks["missing_project_id"].sum()),
            "E3_missing_story_points": int(masks["missing_story_points"].sum()),
            "E4_story_point_parse_failure": int(
                masks["story_point_parse_failure"].sum()
            ),
            "E5_non_finite_story_points": int(
                masks["non_finite_story_points"].sum()
            ),
            "E6_negative_story_points": int(
                masks["negative_story_points"].sum()
            ),
            "E7_zero_story_points": int(zero_sp_mask.sum()),
            "E8_empty_text": int(masks["empty_text"].sum()),
        },

        # --- Outlier info ---
        "story_point_99th_percentile": (
            float(p99) if p99 is not None else None
        ),
        "outlier_note": (
            "Records at or above the 99th percentile are saved to "
            "potential_outliers.csv for review. They are NOT removed "
            "from issues_clean.csv or the split files."
        ),

        # --- Split summary ---
        "split_strategy": "specified_source_target_pair",
        "n_splits": len(split_records),
        "splits": [
            {
                "pair": r["pair"],
                "source_project_id": r["source_project_id"],
                "target_project_id": r["target_project_id"],
                "development_records": r["development_records"],
                "target_test_records": r["target_test_records"],
            }
            for r in split_records
        ],

        # --- Policy notes ---
        "exclusion_notes": {
            "E3": (
                "Story-point NULL: issue was not estimated; "
                "cannot be used as a training or test example."
            ),
            "E4": (
                "Non-numeric story-point string (e.g. 'TBD'): "
                "cannot be cast to a regression target."
            ),
            "E5": (
                "Infinite story-point value: arithmetically invalid."
            ),
            "E6": (
                "Negative story-point value: meaningless for effort "
                "estimation."
            ),
            "E7": (
                "Zero story-point value: conventionally used to flag "
                "'not estimated' rather than 'zero effort'. Excluded "
                "from modelling to avoid label noise. Revise this rule "
                "if the project convention differs."
            ),
            "E8": (
                "Empty text after normalisation: no textual features "
                "available for the SBERT encoder."
            ),
        },
    }


# ============================================================
# 11. MAIN
# ============================================================

def main() -> None:
    # Ensure UTF-8 output on Windows consoles that default to cp1252.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    random_seed: int = args.random_seed

    dry_run: bool = args.dry_run

    # Resolve paths.
    input_path: Path = (
        args.input
        if args.input is not None
        else ROOT_DIR / "data" / "processed" / "issues_raw.csv"
    )
    output_dir: Path = ROOT_DIR / "data" / "processed"
    splits_dir: Path = output_dir / "splits"

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: Load ----------------------------------------
    print("\n========== STEP 1: LOAD ==========")
    df = load_raw(input_path)
    initial_rows = len(df)

    # ---- Step 2: Schema validation ---------------------------
    print("\n========== STEP 2: SCHEMA ==========")
    validate_schema(df)

    # ---- Step 3: Normalise identifiers -----------------------
    print("\n========== STEP 3: NORMALISE IDENTIFIERS ==========")
    df = normalise_identifiers(df)

    # ---- Step 4: Duplicate detection -------------------------
    print("\n========== STEP 4: DUPLICATE DETECTION ==========")
    _, duplicate_count = detect_duplicates(df, output_dir, dry_run)

    # ---- Step 5: Text normalisation --------------------------
    print("\n========== STEP 5: TEXT NORMALISATION ==========")
    df["text"] = df["text"].map(clean_text)
    print(f"  Text normalisation applied to {len(df):,} records.")

    # ---- Step 6: Exclusion masks -----------------------------
    print("\n========== STEP 6: EXCLUSION RULES ==========")
    masks = compute_exclusion_masks(df)

    for rule, mask in masks.items():
        print(f"  {rule}: {int(mask.sum()):,} records affected.")

    # ---- Step 7: Apply exclusions ----------------------------
    print("\n========== STEP 7: APPLY EXCLUSIONS ==========")
    clean_df, exclude_any, zero_sp_mask = apply_exclusions(df, masks)

    print(f"  Records before cleaning : {initial_rows:,}")
    print(f"  Records after cleaning  : {len(clean_df):,}")
    print(f"  Records excluded        : {int(exclude_any.sum()):,}")

    # ---- Step 8: Outlier detection and removal ----------------
    print("\n========== STEP 8: OUTLIER DETECTION AND REMOVAL ==========")

    outlier_df, p99 = flag_outliers(clean_df)

    print(
        f"  99th-percentile story-point threshold : "
        f"{p99 if p99 is not None else 'N/A'}"
    )

    print(
        f"  Records flagged as potential outliers : {len(outlier_df):,}"
    )

    # Save outliers before removing them.
    if not dry_run:
        outlier_df.to_csv(
            output_dir / "potential_outliers.csv",
            index=False
        )

    # Remove flagged records.
    clean_df, removed_outliers = remove_outliers(
        clean_df,
        outlier_df
    )

    print(f"  Records removed as outliers            : {removed_outliers:,}")
    print(f"  Records remaining after removal       : {len(clean_df):,}")

    # ---- Step 9: Summaries -----------------------------------
    print("\n========== STEP 9: PROJECT SUMMARY ==========")
    proj_summary_df = project_summary(clean_df)
    sp_dist_df = story_point_distribution(clean_df)
    print(proj_summary_df.to_string(index=False))

    # ---- Step 10: LOPO splits --------------------------------
    print("\n========== STEP 10: LOPO SPLITS ==========")
    pairs_file = ROOT_DIR / "config" / "project_pairs.csv"

    split_records = create_pair_splits(
        clean_df=clean_df,
        splits_dir=splits_dir,
        pairs_file=pairs_file,
        dry_run=dry_run,
    )

    # ---- Step 11: Build report -------------------------------
    report = build_cleaning_report(
        initial_rows=initial_rows,
        df=df,
        masks=masks,
        exclude_any=exclude_any,
        zero_sp_mask=zero_sp_mask,
        clean_df=clean_df,
        duplicate_count=duplicate_count,
        p99=p99,
        split_records=split_records,
        random_seed=random_seed,
    )

    # ---- Step 12: Save all outputs ---------------------------
    if dry_run:
        print("\n[DRY RUN] File output skipped.")
        print("Report preview:")
        print(json.dumps(report, indent=4))
    else:
        print("\n========== STEP 12: SAVE OUTPUTS ==========")

        clean_df.to_csv(output_dir / "issues_clean.csv", index=False)
        print(f"  Saved: issues_clean.csv  ({len(clean_df):,} rows)")

        proj_summary_df.to_csv(
            output_dir / "project_summary_after_cleaning.csv", index=False
        )
        print("  Saved: project_summary_after_cleaning.csv")

        sp_dist_df.to_csv(
            output_dir / "story_point_distribution.csv", index=False
        )
        print("  Saved: story_point_distribution.csv")

        outlier_df.to_csv(
            output_dir / "potential_outliers.csv", index=False
        )
        print(f"  Saved: potential_outliers.csv  ({len(outlier_df):,} rows)")

        with open(
            output_dir / "cleaning_report.json", "w", encoding="utf-8"
        ) as fh:
            json.dump(report, fh, indent=4)
        print("  Saved: cleaning_report.json")

    # ---- Final summary ---------------------------------------
    print("\n========== FINAL SUMMARY ==========")
    print(f"  Input records              : {initial_rows:,}")
    print(f"  Clean records              : {len(clean_df):,}")
    print(f"  Excluded (hard rules E1–E6, E8) : "
          f"{int((exclude_any & ~zero_sp_mask).sum()):,}")
    print(f"  Excluded (zero SP, rule E7): {int(zero_sp_mask.sum()):,}")
    print(f"  Projects retained          : {clean_df['project_id'].nunique()}")
    print(f"  LOPO splits created        : {len(split_records)}")
    print(f"  Random seed                : {random_seed}")
    print()
    print("Member 2 — data preparation complete.")
    print(
        "Next step: run src/03_encode_text.py (Member 3) to encode "
        "split data with SBERT."
    )


if __name__ == "__main__":
    main()
