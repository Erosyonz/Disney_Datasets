"""
Apriori analysis of Disney genres (revised after the dataset review).

Comment tags used below:
  [KEPT]    same logic as your previous version
  [CHANGED] same place, different behaviour (reason given)
  [ADDED]   new in this version (reason given)
  [REMOVED] taken out (reason given)
  [MOVED]   same code, new position in the file

Summary of the changes:
  1. [ADDED]   One settings block: minsupp 0.15, confidence 0.6, lift 1.2.
  2. [ADDED]   Apriori now mines only films with 2+ genres (one-genre films
               cannot form a rule and inflate lift), minus the 3 films whose
               genres belong to another version of the film.
  3. [CHANGED] Validation and cross-feature sections use that same film set.
  4. [CHANGED] minsupp sensitivity table now shows films needed and rule counts.
  5. [CHANGED] Fragile code fixed: bare except, try/except that never fired,
               max() on an empty set, division by zero on empty test set.
"""

import ast
import math
from itertools import combinations
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option("display.width", 200)

# =====================  SETTINGS  =====================
# [ADDED] All thresholds live here so slides and code always agree.
minsupp = 0.15          # [CHANGED] kept at 0.15 = ~22 films on the 2+ genre set
minsupp_x = 0.10        # [KEPT] cross-feature baskets have rarer items
MIN_CONF = 0.6          # [CHANGED] was 0.5 -> the old filter removed very little
MIN_LIFT = 1.2          # [CHANGED] was 1.0 -> lift 1.0 means "no pattern"
MIN_GENRES = 2          # [ADDED] a film needs 2+ genres to be able to form a rule

# [ADDED] Films whose genre list belongs to a different version of the film.
# Matched on (lowercase title, release year). Edit this list if you find more.
WRONG_VERSIONS = {
    ("beauty and the beast", 1991),
    ("the jungle book", 1994),
    ("cinderella", 2015),
}

# Load the dedicated Apriori files so your other test files remain untouched
train_raw = pd.read_csv("Datasets/disney_train_clean.csv")
train_raw["split"] = "train"

test_raw = pd.read_csv("Datasets/disney_test_clean.csv")
test_raw["split"] = "test"

# Combine train and test into your merged dataframe
merged = pd.concat([train_raw, test_raw], ignore_index=True)
merged.to_csv("disney_merged.csv", index=False)


# =====================  FUNCTIONS  =====================
def support(itemset, tx):
    # [CHANGED] guard against an empty transaction list (division by zero)
    if not tx:
        return 0.0
    s = set(itemset)
    return sum(1 for t in tx if s.issubset(t)) / len(tx)

def apriori_manual(tx, minsupp, verbose=True):
    frequent = {}
    all_items = sorted({i for t in tx for i in t})

    # ---- k = 1 ----
    rows, Lk = [], []
    for item in all_items:
        s = support({item}, tx)
        keep = s >= minsupp
        rows.append({"Itemset": "{" + item + "}", "Support": round(s, 3), "Status": "Keep" if keep else "Drop"})
        if keep:
            frequent[frozenset([item])] = s
            Lk.append(frozenset([item]))
    if verbose:
        print("=== Table k=1 ==="); print(pd.DataFrame(rows).to_string(index=False))

    # ---- k >= 2 ----
    k = 2
    while Lk:
        prev = set(Lk)
        items = sorted(set().union(*Lk))
        cands = [frozenset(c) for c in combinations(items, k)
                 if all(frozenset(sub) in prev for sub in combinations(c, k - 1))]
        if not cands:
            if verbose:
                print(f"\nNo candidates for k={k} -> growing terminates.")
            break
        rows, Lk = [], []
        for c in cands:
            s = support(c, tx)
            keep = s >= minsupp
            rows.append({"Itemset": "{" + ", ".join(sorted(c)) + "}", "Support": round(s, 3), "Status": "Keep" if keep else "Drop"})
            if keep:
                frequent[c] = s
                Lk.append(c)
        if verbose:
            print(f"\n=== Table k={k} ==="); print(pd.DataFrame(rows).to_string(index=False))
        if not Lk:
            if verbose:
                print(f"\nNo itemset at k={k} reached minsupp -> growing terminates.")
            break
        k += 1
    return frequent

def parse_basket(v):
    if pd.isna(v):
        return set()
    v = str(v).strip()
    # If it's a string representation of a list (e.g., "['Animation', 'Comedy']")
    if v.startswith("["):
        try:
            return {g.lower().strip() for g in ast.literal_eval(v)}
        except (ValueError, SyntaxError):   # [CHANGED] was a bare "except:"
            return set()
    # If it's a plain comma-separated string (e.g., "Animation, Comedy")
    elif "," in v:
        return {g.lower().strip() for g in v.split(",")}
    # If it's just a single genre string (e.g., "Animation")
    return {v.lower().strip()}

# [MOVED] generate_rules used to sit further down; it is now defined here so the
# minsupp sensitivity table can also count rules.
def generate_rules(frequent, single_consequent=True):
    rows = []
    for itemset, s_xy in frequent.items():
        if len(itemset) < 2:
            continue
        for r in range(1, len(itemset)):
            for ante in combinations(sorted(itemset), r):
                ante = frozenset(ante); cons = itemset - ante
                if single_consequent and len(cons) != 1:
                    continue
                s_x, s_y = frequent[ante], frequent[cons]
                rows.append({
                    "antecedent": ante, "consequent": cons,
                    "Rule": "{" + ", ".join(sorted(ante)) + "} => {" + ", ".join(sorted(cons)) + "}",
                    "Support": s_xy,
                    "Confidence": s_xy / s_x,
                    "Lift": s_xy / (s_x * s_y),
                    "Leverage": s_xy - s_x * s_y})
    if not rows:
        return pd.DataFrame(columns=["antecedent", "consequent", "Rule", "Support", "Confidence", "Lift", "Leverage"])
    return pd.DataFrame(rows).sort_values("Lift", ascending=False).reset_index(drop=True)


# =====================  BASKETS & FILM SELECTION  =====================
# Find the correct genre column dynamically
genre_col = "genre" if "genre" in merged.columns else ("genres" if "genres" in merged.columns else None)

if genre_col:
    merged["genre_basket"] = merged[genre_col].map(parse_basket)
else:
    raise KeyError("Could not find a 'genre' or 'genres' column in your dataset!")

# [ADDED] Helper columns to detect the wrong-version films.
def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

title_col = find_col(merged, ["title", "movie_title", "name", "movie"])
year_col = find_col(merged, ["year", "disney_year", "release_year"])
if year_col:
    merged["_year"] = pd.to_numeric(merged[year_col], errors="coerce")
elif "release_date" in merged.columns:
    merged["_year"] = pd.to_datetime(merged["release_date"], errors="coerce").dt.year
else:
    merged["_year"] = float("nan")

# [ADDED] Flag the 3 wrong-version films (title AND year must match).
is_wrong = pd.Series(False, index=merged.index)
if title_col and merged["_year"].notna().any():
    norm_titles = merged[title_col].astype(str).str.lower().str.strip()
    for t, y in WRONG_VERSIONS:
        is_wrong |= (norm_titles == t) & (merged["_year"] == y)
else:
    print("WARNING: could not find a title/year column, wrong-version films were NOT removed.")

# [ADDED] Document the film selection so the data-prep slide can quote it.
has_genre = merged["genre_basket"].map(len) > 0
n_genres = merged["genre_basket"].map(len)
print("=== Film selection for Apriori ===")
print("All films                         :", len(merged))
print("Films with a genre list           :", int(has_genre.sum()))
print("  with only 1 genre (excluded)    :", int((n_genres == 1).sum()))
print(f"  with {MIN_GENRES}+ genres                  :", int((n_genres >= MIN_GENRES).sum()))
print("  wrong-version films removed     :", int((is_wrong & (n_genres >= MIN_GENRES)).sum()),
      f"(expected {len(WRONG_VERSIONS)})")

# [ADDED] One-genre films are kept for descriptive statistics, just not for mining.
all_labels = pd.Series([g for b in merged["genre_basket"] for g in b]).value_counts()
print("Top genre labels on ALL films (descriptive):", {k: int(v) for k, v in all_labels.head(8).items()})

# [CHANGED] mining_df replaces "merged" as the film set for everything below.
mining_df = merged[(n_genres >= MIN_GENRES) & ~is_wrong].copy()

transactions = list(mining_df["genre_basket"])
items = sorted({i for t in transactions for i in t})
N = len(transactions)
if N == 0:
    raise ValueError("No films left to mine. Check MIN_GENRES and the genre column.")

print(f"\n|T| = {N} transactions, |I| = {len(items)} items")
for t in transactions[:5]:
    print(sorted(t))

print("Number of transactions (films):", N)
print("Average genres per film       :", round(sum(len(t) for t in transactions) / N, 2))
print(f"Minimum films per frequent itemset at minsupp={minsupp}: {math.ceil(minsupp * N)}")

# =====================  RUN APRIORI ON GENRES  =====================
# [CHANGED] The sensitivity table now shows films needed and the rule count,
# so the choice of 0.15 can be defended with numbers instead of preference.
print("\n=== minsupp sensitivity ===")
print(f"{'minsupp':>8} {'films':>6} {'itemsets':>9} {'max k':>6} {'rules':>6}")
for ms in [0.05, 0.10, 0.15, 0.20, 0.30]:
    f = apriori_manual(transactions, ms, verbose=False)
    r = generate_rules(f)
    n_r = len(r[(r["Confidence"] >= MIN_CONF) & (r["Lift"] >= MIN_LIFT)]) if not r.empty else 0
    max_k = max((len(s) for s in f), default=0)   # [CHANGED] default=0 avoids a crash on an empty result
    print(f"{ms:>8.2f} {math.ceil(ms * N):>6d} {len(f):>9d} {max_k:>6d} {n_r:>6d}")

print()
frequent = apriori_manual(transactions, minsupp)

multi = sorted((s for s in frequent if len(s) >= 2), key=lambda s: (len(s), sorted(s)))
for s in multi:
    print(sorted(s), round(frequent[s], 3))

# =====================  STEP 4: CANDIDATES FOR RULES  =====================
# [KEPT] Lecture-style demo of every split of one itemset.
def rule_candidates(target, tx):
    items = sorted(target)
    s_xy = support(items, tx)
    rows = []
    for r in range(0, len(items) + 1):
        for ante in combinations(items, r):
            ante = set(ante); cons = set(items) - ante
            s_x, s_y = support(ante, tx), support(cons, tx)
            rows.append({
                "Rule": ("{" + ", ".join(sorted(ante)) + "}" if ante else "Ø") + " => " +
                        ("{" + ", ".join(sorted(cons)) + "}" if cons else "Ø"),
                "Confidence": round(s_xy / s_x, 2) if s_x > 0 else 0.0,
                "Lift": round(s_xy / (s_x * s_y), 2) if (s_x * s_y) > 0 else 0.0,
                "Leverage": round(s_xy - s_x * s_y, 2),
                "Kept?": "Yes" if ante and len(cons) == 1 else "No"})
    return pd.DataFrame(rows)

# [CHANGED] The old try/except KeyError could never fire (nothing raised it).
# An explicit membership check does what the message promised.
demo_set = frozenset({"animation", "comedy", "family", "fantasy"})
if demo_set in frequent:
    print(rule_candidates(demo_set, transactions).to_string(index=False))
else:
    print("Target combination not present in frequent itemsets for this minsupp.")

# =====================  RULES FOR ALL FREQUENT ITEMSETS  =====================
# [MOVED] generate_rules is defined in the FUNCTIONS block above.
# [CHANGED] MIN_CONF / MIN_LIFT now come from the SETTINGS block (0.6 / 1.2).
all_rules = generate_rules(frequent)
if all_rules.empty:
    rules_f = all_rules.copy()
else:
    rules_f = all_rules[(all_rules["Confidence"] >= MIN_CONF) & (all_rules["Lift"] >= MIN_LIFT)].reset_index(drop=True)

print(f"Rules generated: {len(all_rules)}   |   after filter (conf >= {MIN_CONF}, lift >= {MIN_LIFT}): {len(rules_f)}")
show = ["Rule", "Support", "Confidence", "Lift", "Leverage"]
if not rules_f.empty:
    print(rules_f[show].round(3).to_string(index=False))

# =====================  VALIDATION  =====================
# [CHANGED] Train/test now come from mining_df (2+ genres, wrong versions removed),
# so validation uses exactly the same films as the main mining.
train_tx = list(mining_df.loc[mining_df["split"] == "train", "genre_basket"])
test_tx = list(mining_df.loc[mining_df["split"] == "test", "genre_basket"])

print(f"Train films: {len(train_tx)} | Test films: {len(test_tx)}")
if len(test_tx) < 50:
    print("Note: the test set is small, so treat 'Holds on test?' as a sanity check, not proof.")

rules_tr = generate_rules(apriori_manual(train_tx, minsupp, verbose=False))
if not rules_tr.empty:
    rules_tr = rules_tr[(rules_tr["Confidence"] >= MIN_CONF) & (rules_tr["Lift"] >= MIN_LIFT)].copy()

def eval_rule(ante, cons, tx):
    s_x, s_y, s_xy = support(ante, tx), support(cons, tx), support(ante | cons, tx)
    conf = s_xy / s_x if s_x > 0 else float("nan")
    lift = s_xy / (s_x * s_y) if s_x > 0 and s_y > 0 else float("nan")
    return conf, lift

if not rules_tr.empty:
    rules_tr[["Test conf", "Test lift"]] = rules_tr.apply(
        lambda r: pd.Series(eval_rule(r["antecedent"], r["consequent"], test_tx)), axis=1
    )
    # [KEPT] "Holds" = still confident enough on test AND better than chance (lift > 1)
    rules_tr["Holds on test?"] = ((rules_tr["Test conf"] >= MIN_CONF) & (rules_tr["Test lift"] > 1)).map({True: "Yes", False: "No"})
    show_cols = ["Rule", "Support", "Confidence", "Lift", "Leverage", "Test conf", "Test lift", "Holds on test?"]
    print(rules_tr[show_cols].round(2).to_string(index=False))
    # [ADDED] One-line summary for the slide
    n_hold = int((rules_tr["Holds on test?"] == "Yes").sum())
    print(f"Rules mined on train that hold on test: {n_hold} of {len(rules_tr)}")
else:
    print("No rules passed the filter on the training films.")

# =====================  CROSS-CHECK WITH mlxtend  =====================
try:
    from mlxtend.frequent_patterns import apriori as mlx_apriori
    all_items = sorted({i for t in transactions for i in t})
    onehot = pd.DataFrame([[item in t for item in all_items] for t in transactions], columns=all_items)
    fi = mlx_apriori(onehot, min_support=minsupp, use_colnames=True)
    mlx = {frozenset(row.itemsets): row.support for row in fi.itertuples()}
    same_sets = set(mlx) == set(frequent)
    max_diff = max(abs(mlx[k] - frequent[k]) for k in frequent) if same_sets and frequent else None
    print("Same as mlxtend:", same_sets)
    print("Max support difference          :", max_diff)
    print("Manual:", len(frequent), "itemsets | mlxtend:", len(mlx), "itemsets")
except ImportError:
    print("mlxtend not installed. Run:  pip install mlxtend  then re-run this cell.")

# =====================  CROSS-FEATURE RULES  =====================
# [CHANGED] Uses mining_df (same films as the genre rules) instead of every film.
# Find the correct gross column dynamically
gross_col = None
for col in ["inflation_adjusted_gross", "total_gross", "worldwide_gross", "gross"]:
    if col in mining_df.columns:
        gross_col = col
        break

if gross_col:
    med_gross = mining_df[gross_col].median()
else:
    med_gross = 0

def make_basket(row):
    genre_vals = row.get("genre_basket")
    if not isinstance(genre_vals, (set, list, frozenset)):
        genre_vals = []
    items = [f"genre:{g}" for g in genre_vals]

    if "mpaa_rating" in row and pd.notna(row["mpaa_rating"]):
        items.append(f"rating:{row['mpaa_rating']}")

    # [CHANGED] uses the helper "_year" column built earlier (same year logic as the film removal)
    year_val = row.get("_year")
    if pd.notna(year_val):
        items.append(f"era:{(int(year_val) // 10) * 10}s")

    if gross_col and pd.notna(row.get(gross_col)):
        items.append("gross:high" if row[gross_col] >= med_gross else "gross:low")

    return set(items)

tx_x = [make_basket(r) for _, r in mining_df.iterrows()]
freq_x = apriori_manual(tx_x, minsupp_x, verbose=False)
print(f"Frequent itemsets with cross-feature baskets (minsupp={minsupp_x}): {len(freq_x)}")

targets = {"gross:high", "gross:low"}
rules_x = generate_rules(freq_x)
if not rules_x.empty:
    rules_x = rules_x[rules_x["consequent"].apply(lambda c: any(t in targets for t in c))]
    rules_x = rules_x[(rules_x["Confidence"] >= MIN_CONF) & (rules_x["Lift"] >= MIN_LIFT)].reset_index(drop=True)
    print(f"Rules predicting gross level (conf >= {MIN_CONF}, lift >= {MIN_LIFT}): {len(rules_x)}")
    if not rules_x.empty:
        print(rules_x[["Rule", "Support", "Confidence", "Lift", "Leverage"]].head(15).round(3).to_string(index=False))

# =====================  VISUALIZATIONS  =====================
if not all_rules.empty and not rules_f.empty:

    # The one number that sums up the graph: the middle Lift of the good rules
    middle_rule_strength = rules_f["Lift"].median()
    print(f"Middle (median) Lift of the good rules: {middle_rule_strength:.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ---------- Left graph: every rule as a dot ----------
    dots = axes[0].scatter(
        all_rules["Support"],
        all_rules["Confidence"],
        c=all_rules["Lift"],
        cmap="viridis",
        s=70,
        edgecolor="k"
    )
    axes[0].axhline(MIN_CONF, ls="--", c="gray", alpha=0.7)
    axes[0].set_xlabel("Support", fontsize=11)
    axes[0].set_ylabel("Confidence ", fontsize=11)
    # [CHANGED] title states the film count so the slide is self-explanatory
    axes[0].set_title(f"Genre Rules: Support vs Confidence", fontsize=12)
    fig.colorbar(dots, ax=axes[0], label="Lift (strength of the pattern)")

    # ---------- Right graph: top 10 strongest rules ----------
    top_rules = rules_f.head(10).iloc[::-1]
    axes[1].barh(
        top_rules["Rule"],
        top_rules["Lift"],
        color="tab:orange",
        edgecolor="k",
        alpha=0.85
    )
    axes[1].axvline(1.0, ls="--", c="gray", alpha=0.7,
                    label="Just luck = 1.0")
    axes[1].axvline(middle_rule_strength, ls=":", c="red",
                    label=f"Middle rule strength = {middle_rule_strength:.2f}")
    axes[1].set_xlabel("Lift", fontsize=11)
    axes[1].set_title(f"Top Genre Rules by Lift", fontsize=12)

    # Legend placed below the chart so it does not cover any bars
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)

    plt.tight_layout()

    # Save the picture in your folder (bbox_inches keeps the legend from being cut off)
    plt.savefig("apriori_genre_rules_analysis.png", dpi=300, bbox_inches="tight")
    print("Graph saved as 'apriori_genre_rules_analysis.png'")

    plt.show()
else:
    # [ADDED] Say why there is no graph instead of silently skipping it
    print("No rules passed the filter, so no graph was drawn. Try a lower minsupp, MIN_CONF or MIN_LIFT.")