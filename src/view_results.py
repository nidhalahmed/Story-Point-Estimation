import pandas as pd

base = "data/processed/splits/pair_38/embeddings"

source = pd.read_csv(f"{base}/development_metadata.csv")
target = pd.read_csv(f"{base}/target_test_metadata.csv")

print("SOURCE: DAEMON")
print(source["story_points"].describe(percentiles=[
    0.5, 0.9, 0.95, 0.99
]))

print("\nTARGET: DATACASS")
print(target["story_points"].describe(percentiles=[
    0.5, 0.9, 0.95, 0.99
]))