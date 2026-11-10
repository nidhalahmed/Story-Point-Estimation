
import os
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, URL

from dotenv import load_dotenv


# ----------------------------------
# 1. Project configuration
# ----------------------------------

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "issues_raw.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------
# 2. Connect to MySQL
# ----------------------------------

connection_url = URL.create(
    drivername="mysql+pymysql",
    username=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST", "localhost"),
    port=int(os.getenv("DB_PORT", "3306")),
    database=os.getenv("DB_NAME", "TAWOS")
)

engine = create_engine(connection_url)


# ----------------------------------
# 3. Extract issue data
# ----------------------------------

query = """
SELECT
    i.ID AS issue_id,
    i.Issue_Key AS issue_key,
    p.Project_Key AS project_id,
    p.Name AS project_name,

    CONCAT_WS(
        '\\n',
        NULLIF(TRIM(i.Title), ''),
        NULLIF(TRIM(i.Description_Text), '')
    ) AS text,

    i.Story_Point AS story_points

FROM Issue i

JOIN Project p
    ON i.Project_ID = p.ID;
"""


def load_data():

    print("Connecting to TAWOS database...")

    df = pd.read_sql_query(query, engine)

    print("Data successfully extracted!")

    return df


# ----------------------------------
# 4. Inspect extracted data
# ----------------------------------

def inspect_data(df):

    print("\n========== DATASET OVERVIEW ==========")

    print("Total records:", len(df))

    print("Unique issue IDs:", df["issue_id"].nunique())

    print("Unique projects:", df["project_id"].nunique())

    print("\n========== MISSING VALUES ==========")

    print(df.isnull().sum())

    print("\n========== DUPLICATE ISSUE IDs ==========")

    print(df["issue_id"].duplicated().sum())

    print("\n========== PROJECT COUNTS ==========")

    print(
        df.groupby(
            ["project_id", "project_name"]
        ).size().sort_values(ascending=False)
    )

    print("\n========== STORY POINT DISTRIBUTION ==========")

    print(
        df["story_points"]
        .value_counts(dropna=False)
        .sort_index()
    )

    print("\n========== SAMPLE RECORDS ==========")

    print(df.head())


# ----------------------------------
# 5. Validate and save
# ----------------------------------

def main():
    try:
        df = load_data()

        required_columns = [
            "issue_id",
            "project_id",
            "text",
            "story_points"
        ]

        missing_columns = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}"
            )

        inspect_data(df)

        df.to_csv(
            OUTPUT_FILE,
            index=False,
            encoding="utf-8"
        )

        print("\n========== EXPORT COMPLETE ==========")

        print("Saved to:", OUTPUT_FILE)

    except Exception as e:

        print("Error:", e)

    finally:

        engine.dispose()


if __name__ == "__main__":
    main()
