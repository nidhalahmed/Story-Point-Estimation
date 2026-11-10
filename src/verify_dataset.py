import pandas as pd

issues = pd.read_csv("data/processed/issues_clean.csv")
pairs = pd.read_csv("config/project_pairs.csv")

# Display actual project names and IDs
projects = (
    issues[["project_id", "project_name"]]
    .drop_duplicates()
    .sort_values("project_name")
)

print("PROJECTS IN DATASET")
print(projects.to_string(index=False))

# Check every source and target name
available_names = set(issues["project_name"].dropna().str.strip())

missing = []

for _, pair in pairs.iterrows():
    for column in ["source_project_id", "target_project_id"]:
        name = str(pair[column]).strip()

        if name not in available_names:
            missing.append((pair["pair"], column, name))

print("\nMISSING PROJECT NAMES")

if missing:
    for item in missing:
        print(item)
else:
    print("All 78 source/target references matched.")

print("\nNumber of pairs:", len(pairs))
print("Unique projects:", issues["project_name"].nunique())