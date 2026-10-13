import ast
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

TRAIN_CSV = "disney_movies_train.csv"
TEST_CSV = "disney_movies_test.csv"
TARGET = "log_adj"          # log10(inflation_adjusted_gross); use "log_total" for total_gross
TOP_GENRES = ["comedy", "family", "drama", "fantasy", "action",
              "animation", "romance", "adventure", "scifi"]


def parse_genres(value):
    if pd.isna(value):
        return []
    value = str(value).strip()
    if value.startswith("["):
        return [g.lower() for g in ast.literal_eval(value)]
    return [value.lower()]


def prepare(path):
    df = pd.read_csv(path)
    df["year"] = pd.to_datetime(df["release_date"]).dt.year
    glist = df["genre"].map(parse_genres)
    for g in TOP_GENRES:
        df[g] = glist.map(lambda lst, g=g: int(g in lst))
    df["rating"] = df["mpaa_rating"].replace("Not Rated", np.nan).fillna("Unknown")
    df = df[(df["total_gross"] > 0) & (df["inflation_adjusted_gross"] > 0)].copy()
    df["log_total"] = np.log10(df["total_gross"])
    df["log_adj"] = np.log10(df["inflation_adjusted_gross"])
    return df.reset_index(drop=True)


train, test = prepare(TRAIN_CSV), prepare(TEST_CSV)
print(f"train rows: {len(train)} | test rows: {len(test)}")

# ----------------------------------------------------- candidate models
rating_term = "C(rating, Treatment('PG'))"
formulas = {
    "Year only": f"{TARGET} ~ year",
    "Year + rating": f"{TARGET} ~ year + {rating_term}",
    "Year + rating + genres": f"{TARGET} ~ year + {rating_term} + " + " + ".join(TOP_GENRES),
}


def metrics(y_true, y_pred, baseline):
    """R^2 vs the training mean, RMSE and MAE in log10 units."""
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - baseline) ** 2).sum()
    return {"R2": 1 - ss_res / ss_tot,
            "RMSE": np.sqrt(((y_true - y_pred) ** 2).mean()),
            "MAE": np.abs(y_true - y_pred).mean()}


baseline = train[TARGET].mean()           # "always guess the average" model
rows = [{"model": "Baseline (mean)",
         **{f"train_{k}": v for k, v in metrics(train[TARGET], baseline, baseline).items()},
         **{f"test_{k}": v for k, v in metrics(test[TARGET], baseline, baseline).items()}}]
fits = {}
for name, f in formulas.items():
    m = smf.ols(f, train).fit()           # fit on TRAIN only
    fits[name] = m
    rows.append({"model": name,
                 **{f"train_{k}": v for k, v in metrics(train[TARGET], m.predict(train), baseline).items()},
                 **{f"test_{k}": v for k, v in metrics(test[TARGET], m.predict(test), baseline).items()}})

results = pd.DataFrame(rows).set_index("model")
pd.set_option("display.width", 160)
print("\n", results.round(3))

best_name = "Year + rating + genres"
best = fits[best_name]
print(f"\n===== {best_name} (fit on train) =====")
print(best.summary().tables[1])

# ------------------------------------------------------------- plots
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# 1. predicted vs actual, train and test
ax = axes[0]
ax.scatter(best.predict(train), train[TARGET], alpha=0.35, s=16, label="train", color="tab:blue")
ax.scatter(best.predict(test), test[TARGET], alpha=0.8, s=22, label="test", color="tab:red")
lo = min(train[TARGET].min(), test[TARGET].min())
hi = max(train[TARGET].max(), test[TARGET].max())
ax.plot([lo, hi], [lo, hi], "k--", lw=1)
ax.set_xlim(6.3, 9.8)
ax.set_xlabel("Predicted log10(adjusted gross)")
ax.set_ylabel("Actual log10(adjusted gross)")
ax.set_title(f"Predicted vs actual\ntest R² = {results.loc[best_name, 'test_R2']:.3f}")
ax.legend()

# 2. regression line on year for train and test
ax = axes[1]
m_year = fits["Year only"]
sns.regplot(data=train, x="year", y=TARGET, ax=ax, ci=95, scatter_kws={"alpha": 0.3, "s": 16},
            line_kws={"color": "black"}, label="train")
ax.scatter(test["year"], test[TARGET], color="tab:red", s=22, alpha=0.8, label="test")
ax.set_xlabel("Release year")
ax.set_ylabel("log10(adjusted gross)")
ax.set_title("Year-only regression (fit on train)\nred = unseen test movies")
ax.legend()

# 3. test R^2 by model
ax = axes[2]
plot_df = results["test_R2"]
colors = ["tab:gray"] + ["tab:green" if v > 0 else "tab:red" for v in plot_df.iloc[1:]]
ax.bar(range(len(plot_df)), plot_df.values, color=colors)
ax.set_xticks(range(len(plot_df)))
ax.set_xticklabels([n.replace(" + ", "\n+ ") for n in plot_df.index], fontsize=9)
ax.axhline(0, color="black", lw=1)
ax.set_ylabel("R² on test set")
ax.set_title("Which model generalizes best?")
for i, v in enumerate(plot_df.values):
    ax.text(i, v, f"{v:.3f}", ha="center", va="bottom" if v >= 0 else "top")

plt.tight_layout()
plt.savefig("disney_train_test.png", dpi=150)
plt.show()