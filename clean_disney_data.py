"""
Clean the merged Disney train/test files so they have ONLY the columns the charts
need and NO missing values.

Usage:  python clean_disney_data.py
Needs:  pip install pandas
Input:  disney_train_merged.csv, disney_test_merged.csv
        (made by disney_merged_regression.py; in this folder or in "Datasets")
Output: disney_train_clean.csv, disney_test_clean.csv, written next to this script
"""
import ast
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
KEEP = ["movie_title", "year", "mpaa_rating", "genre", "budget", "runtime", "total_gross"]


def find_csv(name):
    for folder in (HERE, HERE / "Datasets"):
        if (folder / name).exists():
            return folder / name
    raise FileNotFoundError(f"Could not find {name} in {HERE} or {HERE / 'Datasets'}")


def standardize_genres(value):
    """Turn "['fantasy', 'family']" or "Adventure" into "fantasy, family" / "adventure"."""
    if pd.isna(value):
        return None
    value = str(value).strip()
    if value.startswith("["):
        items = [g.strip().lower() for g in ast.literal_eval(value)]
    else:
        items = [value.lower()]
    items = [g for g in items if g]
    return ", ".join(items) if items else None


def clean(path, label):
    df = pd.read_csv(path)
    start = len(df)
    log = {}

    # 1. keep only films that were found in movies.csv (they have a budget and runtime)
    before = len(df)
    df = df.dropna(subset=["budget", "runtime"])
    log["no match / no budget or runtime"] = before - len(df)

    # 2. remove impossible values
    before = len(df)
    df = df[(df["total_gross"] > 0) & (df["budget"] > 0)]
    log["zero gross or budget"] = before - len(df)

    # 3. standardize genres to one format, then drop films with no genre
    df = df.copy()
    df["genre"] = df["genre"].map(standardize_genres)
    before = len(df)
    df = df.dropna(subset=["genre"])
    log["blank genre"] = before - len(df)

    # 4. drop films with no rating
    before = len(df)
    df = df.dropna(subset=["mpaa_rating"])
    df = df[df["mpaa_rating"].str.strip() != ""]
    df = df[df["mpaa_rating"] != "Not Rated"]
    log["blank or 'Not Rated' rating"] = before - len(df)

    # 5. keep only the needed columns
    df = df[KEEP].reset_index(drop=True)
    df["year"] = df["year"].astype(int)

    assert df.isna().sum().sum() == 0, "There are still missing values!"
    print(f"\n{label}: {start} rows in -> {len(df)} rows out")
    for reason, n in log.items():
        print(f"   removed {n:>3} for: {reason}")
    return df


train = clean(find_csv("disney_train_merged.csv"), "TRAIN")
test = clean(find_csv("disney_test_merged.csv"), "TEST")

train.to_csv(HERE / "disney_train_clean.csv", index=False)
test.to_csv(HERE / "disney_test_clean.csv", index=False)
print(f"\nSaved {len(train)} train and {len(test)} test rows. Columns: {', '.join(KEEP)}")
print("Missing values left:", int(train.isna().sum().sum() + test.isna().sum().sum()))