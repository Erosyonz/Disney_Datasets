"""
Plain-language charts and a readable data table for the merged Disney dataset
(Disney movies + budget/runtime/score from the general movies.csv).

Usage:  python disney_readable_charts.py
Needs:  pip install pandas numpy statsmodels matplotlib
Input:  disney_train_clean.csv and disney_test_clean.csv
        (created by clean_disney_data.py; in this folder or in "Datasets")
Output: a "charts" folder with 5 PNGs, plus disney_movies_readable.csv
"""
import textwrap
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, FixedLocator, NullLocator
import statsmodels.formula.api as smf

HERE = Path(__file__).resolve().parent
OUT = HERE / "charts"
OUT.mkdir(exist_ok=True)


def find_csv(name):
    for folder in (HERE, HERE / "Datasets"):
        if (folder / name).exists():
            return folder / name
    raise FileNotFoundError(f"Could not find {name} in {HERE} or {HERE / 'Datasets'}")


# ------------------------------------------------------------------ style
BLUE, ORANGE, GREEN, RED, GREY = "#2b6cb0", "#ed8936", "#2f9e44", "#d9480f", "#adb5bd"
DARK = "#212529"
plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 18, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#868e96", "axes.labelcolor": DARK, "text.color": DARK,
    "xtick.color": DARK, "ytick.color": DARK, "figure.dpi": 100,
    "axes.grid": True, "grid.color": "#e9ecef", "axes.axisbelow": True,
})


def money(v, _=None):
    """Format dollars the way a person would say them: $25M, $1.2B."""
    return f"${v / 1e9:.1f}B" if v >= 1e9 else f"${v / 1e6:,.0f}M"


def start_figure(title, subtitle, size=(11, 6.5)):
    fig, ax = plt.subplots(figsize=size)
    fig.suptitle(title, x=0.01, y=0.985, ha="left", fontsize=20, fontweight="bold")
    fig.text(0.01, 0.905, subtitle, ha="left", va="top", fontsize=13, color="#495057")
    return fig, ax


def finish(fig, name, note, left=0.11, bottom=0.20):
    fig.text(0.01, 0.012, textwrap.fill(note, 135), ha="left", va="bottom", fontsize=9.5, color="#868e96")
    fig.subplots_adjust(top=0.80, bottom=bottom, left=left, right=0.97)
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------- data
def parse_genres(v):
    """Clean files store genres like "comedy, drama" (one standard format)."""
    if pd.isna(v):
        return []
    return [g.strip().lower() for g in str(v).split(",") if g.strip()]


GENRES = ["comedy", "family", "drama", "fantasy", "action",
          "animation", "romance", "adventure", "scifi"]

train_raw = pd.read_csv(find_csv("disney_train_clean.csv"))
test_raw = pd.read_csv(find_csv("disney_test_clean.csv"))


def prepare(d):
    d = d.copy()
    gl = d["genre"].map(parse_genres)
    for g in GENRES:
        d[g] = gl.map(lambda lst, g=g: int(g in lst))
    d["rating"] = d["mpaa_rating"].replace("Not Rated", np.nan).fillna("Unknown")
    d = d.dropna(subset=["budget", "runtime"])
    d = d[(d["total_gross"] > 0) & (d["budget"] > 0)]
    d["log_gross"] = np.log10(d["total_gross"])
    d["log_budget"] = np.log10(d["budget"])
    return d.reset_index(drop=True)


train, test = prepare(train_raw), prepare(test_raw)
films = pd.concat([train, test], ignore_index=True)
n_films, y0, y1 = len(films), int(films["year"].min()), int(films["year"].max())
NOTE = (f"Based on {n_films} Disney films released {y0}-{y1} that have budget data. "
        "Box office = US/domestic, not adjusted for inflation. Sources: Kaggle Disney and IMDb movie datasets.")

# ------------------------------------------------------------ models
terms = "year + C(rating, Treatment('PG')) + " + " + ".join(GENRES)
f_basic = f"log_gross ~ {terms}"
f_full = f"log_gross ~ {terms} + log_budget + runtime"
m_basic = smf.ols(f_basic, train).fit()
m_full = smf.ols(f_full, train).fit()
base = train["log_gross"].mean()


def r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - base) ** 2).sum()


r2_basic = r2(test["log_gross"], m_basic.predict(test))
r2_full = r2(test["log_gross"], m_full.predict(test))
pred = 10 ** m_full.predict(test)
ratio = test["total_gross"] / pred
within2 = ((ratio < 2) & (ratio > 0.5)).mean()
within3 = ((ratio < 3) & (ratio > 1 / 3)).mean()

# ------------------------------------------- CHART 1: budget groups
fig, ax = start_figure("Bigger budgets bring bigger box office",
                       "Typical (median) budget and box office for Disney films in each budget group")
bins = [0, 25e6, 50e6, 100e6, 150e6, 1e12]
labels = ["Under $25M", "$25M–50M", "$50M–100M", "$100M–150M", "Over $150M"]
films["group"] = pd.cut(films["budget"], bins, labels=labels)
g = films.groupby("group", observed=True).agg(budget=("budget", "median"),
                                              gross=("total_gross", "median"),
                                              n=("budget", "size"))
x = np.arange(len(g))
b1 = ax.bar(x - 0.2, g["budget"], 0.4, color=GREY, label="Typical budget (cost to make)")
b2 = ax.bar(x + 0.2, g["gross"], 0.4, color=BLUE, label="Typical box office (money earned)")
for bars in (b1, b2):
    for r in bars:
        ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 4e6, money(r.get_height()),
                ha="center", fontsize=12, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([f"{lab}\n({n} films)" for lab, n in zip(g.index, g["n"])])
ax.yaxis.set_major_formatter(FuncFormatter(money))
ax.set_ylabel("")
ax.grid(axis="x", visible=False)
ax.legend(frameon=False, loc="upper left", fontsize=12)
ax.set_xlabel("Budget group")
finish(fig, "1_budget_groups.png", NOTE)

# ---------------------------------- CHART 2: budget vs earnings
fig, ax = start_figure("Every film: what it cost vs what it earned",
                       "Each dot is one Disney film. The black line is the overall trend.",
                       size=(11, 7.2))
trend = smf.ols("log_gross ~ log_budget", films).fit()
ax.scatter(films["budget"], films["total_gross"], s=38, color=BLUE, alpha=0.45, edgecolor="none")
xs = np.linspace(films["log_budget"].min(), films["log_budget"].max(), 50)
ax.plot(10 ** xs, 10 ** trend.predict(pd.DataFrame({"log_budget": xs})), color="black", lw=2.5,
        label="Overall trend")
lim = [2e6, 1e9]
ax.plot(lim, lim, color=RED, ls="--", lw=1.5, label="Break-even line (earned = budget)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(2.5e6, 4e8)
ax.set_ylim(5e5, 1e9)
ax.xaxis.set_major_locator(FixedLocator([5e6, 10e6, 25e6, 50e6, 100e6, 250e6]))
ax.yaxis.set_major_locator(FixedLocator([1e6, 10e6, 100e6, 1e9]))
ax.xaxis.set_minor_locator(NullLocator())
ax.yaxis.set_minor_locator(NullLocator())
ax.xaxis.set_major_formatter(FuncFormatter(money))
ax.yaxis.set_major_formatter(FuncFormatter(money))
ax.set_xlabel("Budget (what it cost to make)")
ax.set_ylabel("Box office (what it earned)")
picks = {  # title: (x offset, y offset in points, text alignment) for the label
    "The Avengers": (-10, 22, "center"), "The Lion King": (-95, 30, "left"),
    "Pretty Woman": (-110, 18, "left"), "John Carter": (-10, -48, "center"),
    "Treasure Planet": (-30, -52, "center"),
}
for title, (dx, dy, align) in picks.items():
    r = films[films["movie_title"] == title]
    if r.empty:
        continue
    r = r.iloc[0]
    ax.scatter([r["budget"]], [r["total_gross"]], s=60, color=ORANGE, edgecolor="black", zorder=5)
    ax.annotate(title, (r["budget"], r["total_gross"]), xytext=(dx, dy), textcoords="offset points",
                fontsize=11.5, fontweight="bold", ha=align,
                arrowprops=dict(arrowstyle="-", color="#495057"))
ax.text(0.98, 0.04, "Dots above the red line made more than they cost.\nDots below it earned less than the budget.",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=11, color="#495057",
        bbox=dict(boxstyle="round", fc="white", ec="#dee2e6"))
ax.legend(frameon=False, loc="upper left", fontsize=11.5)
finish(fig, "2_budget_vs_earnings.png",
       NOTE + " Both axes use a multiplying scale, so each step is a big jump.")

# ----------------------- CHART 3: how much can we explain?
fig, ax = start_figure("How much of the difference in earnings can we explain?",
                       "Tested on films the model had never seen (higher = more of the story explained)",
                       size=(11, 5.6))
rows = [("Just guessing\nthe average", 0.0, GREY),
        ("Release year,\nrating and genre", max(r2_basic, 0) * 100, ORANGE),
        ("Those, plus budget\nand runtime", r2_full * 100, BLUE)]
y = np.arange(len(rows))[::-1]
for yi, (lab, val, col) in zip(y, rows):
    ax.barh(yi, 100, color="#f1f3f5", height=0.55)
    ax.barh(yi, val, color=col, height=0.55)
    ax.text(max(val, 0) + 1.5, yi, f"{val:.0f}%", va="center", fontsize=15, fontweight="bold")
ax.set_yticks(y)
ax.set_yticklabels([r[0] for r in rows], fontsize=13)
ax.set_xlim(0, 100)
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.set_xlabel("Share of the difference in box office that is explained")
ax.grid(False)
fig.text(0.01, 0.105, "The rest comes from things not in the data: marketing, sequels and franchises, "
         "reviews, competing releases.", ha="left", va="bottom", fontsize=11.5, color="#495057")
finish(fig, "3_how_much_explained.png", NOTE, left=0.30, bottom=0.27)

# --------------------------- CHART 4: prediction accuracy
fig, ax = start_figure("How close are the predictions?",
                       f"{within2:.0%} of unseen films earned within 2x of the prediction "
                       f"({within3:.0%} within 3x)", size=(10, 7.4))
lim = np.array([3e6, 7e8])
ax.fill_between(lim, lim / 2, lim * 2, color=GREEN, alpha=0.12, label="Within 2x of the prediction")
ax.plot(lim, lim, color="black", lw=1.5, ls="--", label="Perfect prediction")
inside = (ratio < 2) & (ratio > 0.5)
ax.scatter(pred[inside], test["total_gross"][inside], s=45, color=GREEN, alpha=0.8, label="Close (within 2x)")
ax.scatter(pred[~inside], test["total_gross"][~inside], s=45, color=ORANGE, alpha=0.9, label="Further off")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(*lim)
ax.set_ylim(*lim)
ticks = [5e6, 10e6, 25e6, 50e6, 100e6, 250e6, 500e6]
for axis in (ax.xaxis, ax.yaxis):
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_minor_locator(NullLocator())
    axis.set_major_formatter(FuncFormatter(money))
ax.set_xlabel("What the model predicted the film would earn")
ax.set_ylabel("What the film actually earned")
ax.legend(frameon=False, loc="upper left", fontsize=11.5)
finish(fig, "4_prediction_accuracy.png", f"{len(test)} test films the model never saw. " + NOTE,)

# ------------------ CHART 5: what changes a film's earnings
p, c = m_full.pvalues, m_full.params


def pct(coef, scale=1.0):
    return (10 ** (coef * scale) - 1) * 100


items = [
    ("Doubling the budget", (2 ** c["log_budget"] - 1) * 100, p["log_budget"]),
    ("R rating (instead of PG)", pct(c["C(rating, Treatment('PG'))[T.R]"]), p["C(rating, Treatment('PG'))[T.R]"]),
    ("Animated film", pct(c["animation"]), p["animation"]),
    ("Action film", pct(c["action"]), p["action"]),
    ("10 extra minutes of runtime", pct(c["runtime"], 10), p["runtime"]),
    ("Releasing 10 years later", pct(c["year"], 10), p["year"]),
]
fig, ax = start_figure("What changes a Disney film's box office?",
                       "Typical change in earnings, comparing otherwise similar films",
                       size=(11, 6.2))
items = items[::-1]
for i, (lab, val, pv) in enumerate(items):
    sig = pv < 0.05
    col = (GREEN if val > 0 else RED) if sig else GREY
    ax.barh(i, val, color=col, height=0.55)
    txt = f"{val:+.0f}%" + ("" if sig else "  (no clear effect)")
    ax.text(val + (2 if val >= 0 else -2), i, txt, va="center",
            ha="left" if val >= 0 else "right", fontsize=12.5, fontweight="bold" if sig else "normal",
            color=DARK if sig else "#6c757d")
ax.set_yticks(range(len(items)))
ax.set_yticklabels([it[0] for it in items], fontsize=13)
ax.axvline(0, color="black", lw=1)
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.0f}%"))
ax.set_xlim(-85, 95)
ax.set_xlabel("Change in box office")
ax.grid(axis="y", visible=False)
ax.text(0.02, 0.04, "Colored bars: effect is clear.\nGrey bars: too uncertain to tell\nfrom no effect at all.",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=10.5, color="#495057",
        bbox=dict(boxstyle="round", fc="white", ec="#dee2e6"))
finish(fig, "5_what_matters.png", NOTE, left=0.30)

# -------------------------------------- readable data table
allfilms = pd.concat([train_raw.assign(Set="Training"), test_raw.assign(Set="Testing")], ignore_index=True)
table = pd.DataFrame({
    "Movie": allfilms["movie_title"],
    "Release year": allfilms["year"],
    "Rating": allfilms["mpaa_rating"],
    "Genres": allfilms["genre"].map(lambda v: ", ".join(x.title() for x in parse_genres(v))),
    "Budget ($ millions)": (allfilms["budget"] / 1e6).round(1),
    "Runtime (minutes)": allfilms["runtime"].astype(int),
    "Box office ($ millions)": (allfilms["total_gross"] / 1e6).round(1),
    "Data split": allfilms["Set"],
})
table = table.sort_values(["Release year", "Movie"]).reset_index(drop=True)
table.to_csv(HERE / "disney_movies_readable.csv", index=False)

print(f"Charts saved to: {OUT}")
print(f"Readable table saved to: {HERE / 'disney_movies_readable.csv'}")
print(f"Explained: basic {r2_basic:.0%}, with budget+runtime {r2_full:.0%} | "
      f"within 2x: {within2:.0%}, within 3x: {within3:.0%}")
for lab, val, pv in items[::-1]:
    print(f"  {lab}: {val:+.0f}%  (p = {pv:.3f})")