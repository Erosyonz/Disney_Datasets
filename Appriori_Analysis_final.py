import ast
from itertools import combinations
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option("display.width", 200)

# Load the dedicated Apriori files so your other test files remain untouched
train_raw = pd.read_csv("Datasets/disney_train_clean.csv")
train_raw["split"] = "train"

test_raw = pd.read_csv("Datasets/disney_test_clean.csv")
test_raw["split"] = "test"

# Combine train and test into your merged dataframe
merged = pd.concat([train_raw, test_raw], ignore_index=True)
merged.to_csv("disney_merged.csv", index=False)

# =====================  APRIORI FUNCTIONS  =====================
def support(itemset, tx):
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
        except:
            return set()
    # If it's a plain comma-separated string (e.g., "Animation, Comedy")
    elif "," in v:
        return {g.lower().strip() for g in v.split(",")}
    # If it's just a single genre string (e.g., "Animation")
    return {v.lower().strip()}

# Find the correct genre column dynamically
genre_col = "genre" if "genre" in merged.columns else ("genres" if "genres" in merged.columns else None)

if genre_col:
    merged["genre_basket"] = merged[genre_col].map(parse_basket)
else:
    raise KeyError("Could not find a 'genre' or 'genres' column in your dataset!")

transactions = [b for b in merged["genre_basket"] if len(b) > 0]
items = sorted({i for t in transactions for i in t})
N = len(transactions)
minsupp = 0.05

print(f"|T| = {N} transactions, |I| = {len(items)} items")
for t in transactions[:5]:
    print(sorted(t))

print("Number of transactions (films):", N)
print("Average genres per film       :", round(sum(len(t) for t in transactions) / N, 2))

# =====================  RUN APRIORI ON GENRES  =====================
for ms in [0.05, 0.10, 0.15, 0.20, 0.30]:
    f = apriori_manual(transactions, ms, verbose=False)
    print(f"minsupp={ms:.2f} -> {len(f):3d} frequent itemsets, largest size k={max(len(s) for s in f)}")
    
frequent = apriori_manual(transactions, minsupp)

multi = sorted((s for s in frequent if len(s) >= 2), key=lambda s: (len(s), sorted(s)))
for s in multi:
    print(sorted(s), round(frequent[s], 3))

# =====================  STEP 4: CANDIDATES FOR RULES  =====================
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

try:
    print(rule_candidates({"animation", "comedy", "family", "fantasy"}, transactions).to_string(index=False))
except KeyError:
    print("Target combination not present in frequent itemsets for this minsupp.")

# =====================  RULES FOR ALL FREQUENT ITEMSETS  =====================
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

MIN_CONF, MIN_LIFT = 0.5, 1.0
all_rules = generate_rules(frequent)
rules_f = all_rules[(all_rules["Confidence"] >= MIN_CONF) & (all_rules["Lift"] >= MIN_LIFT)].reset_index(drop=True)

print(f"Rules generated: {len(all_rules)}   |   after filter (conf >= {MIN_CONF}, lift >= {MIN_LIFT}): {len(rules_f)}")
show = ["Rule", "Support", "Confidence", "Lift", "Leverage"]
if not rules_f.empty:
    print(rules_f[show].round(3).to_string(index=False))

# =====================  VALIDATION  =====================
train_df = merged[merged["split"] == "train"]
test_df = merged[merged["split"] == "test"]
train_tx = [parse_basket(b) for b in train_df["genre_basket"]]
test_tx = [parse_basket(b) for b in test_df["genre_basket"]]

print(f"Train films: {len(train_tx)} | Test films: {len(test_tx)}")

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
    rules_tr["Holds on test?"] = ((rules_tr["Test conf"] >= MIN_CONF) & (rules_tr["Test lift"] > 1)).map({True: "Yes", False: "No"})
    show_cols = ["Rule", "Support", "Confidence", "Lift", "Leverage", "Test conf", "Test lift", "Holds on test?"]
    print(rules_tr[show_cols].round(2).to_string(index=False))

# =====================  CROSS-CHECK WITH mlxtend  =====================
try:
    from mlxtend.frequent_patterns import apriori as mlx_apriori
    all_items = sorted({i for t in transactions for i in t})
    onehot = pd.DataFrame([[item in t for item in all_items] for t in transactions], columns=all_items)
    fi = mlx_apriori(onehot, min_support=minsupp, use_colnames=True)
    mlx = {frozenset(row.itemsets): row.support for row in fi.itertuples()}
    same_sets = set(mlx) == set(frequent)
    max_diff = max(abs(mlx[k] - frequent[k]) for k in frequent) if same_sets else None
    print("Same as mlxtend:", same_sets)
    print("Max support difference          :", max_diff)
    print("Manual:", len(frequent), "itemsets | mlxtend:", len(mlx), "itemsets")
except ImportError:
    print("mlxtend not installed. Run:  pip install mlxtend  then re-run this cell.")

# =====================  CROSS-FEATURE RULES  =====================
# Find the correct gross column dynamically
gross_col = None
for col in ["inflation_adjusted_gross", "total_gross", "worldwide_gross", "gross"]:
    if col in merged.columns:
        gross_col = col
        break

if gross_col:
    med_gross = merged[gross_col].median()
else:
    med_gross = 0

def make_basket(row):
    genre_vals = row.get("genre_basket")
    if pd.isna(genre_vals):
        genre_vals = []
    items = [f"genre:{g}" for g in genre_vals]
    
    if "mpaa_rating" in row and pd.notna(row["mpaa_rating"]):
        items.append(f"rating:{row['mpaa_rating']}")
        
    year_val = row.get("year") or row.get("disney_year")
    if pd.notna(year_val):
        try:
            items.append(f"era:{(int(year_val) // 10) * 10}s")
        except:
            pass
            
    if gross_col and pd.notna(row.get(gross_col)):
        items.append("gross:high" if row[gross_col] >= med_gross else "gross:low")
        
    return set(items)

tx_x = [make_basket(r) for _, r in merged.iterrows()]
minsupp_x = 0.10
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
    axes[0].set_xlabel("Support (how common the genre mix is)", fontsize=11)
    axes[0].set_ylabel("Confidence (how often the rule is right)", fontsize=11)
    axes[0].set_title("Genre Rules: Support vs Confidence", fontsize=12)
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
    axes[1].set_xlabel("Lift (1.0 = no better than chance)", fontsize=11)
    axes[1].set_title(f"Top Genre Rules by Lift (Showing {len(top_rules)} rules)", fontsize=12)

    # Legend placed below the chart so it does not cover any bars
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)

    plt.tight_layout()

    # Save the picture in your folder (bbox_inches keeps the legend from being cut off)
    plt.savefig("apriori_genre_rules_analysis.png", dpi=300, bbox_inches="tight")
    print("Graph saved as 'apriori_genre_rules_analysis.png'")

    plt.show()