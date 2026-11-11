import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
C = {"flat": "#2a78d6", "graph": "#eb6834", "hybrid": "#1baf7a", "calls": "#eda100"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "figure.facecolor": SURF, "axes.facecolor": SURF})

fr = pd.read_csv("results/frontier.csv", index_col=0)
co = pd.read_csv("results/completion.csv", index_col=0)
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)


def draw(ax, df, series, title, xlabel, dy=None):
    dy = dy or {}
    for key, (row, label) in series.items():
        x = [int(c) for c in df.columns]; y = df.loc[row].values
        ax.plot(x, y, color=C[key], lw=2, marker="o", ms=8, mec=SURF, mew=2, zorder=3)
        ax.annotate(label, (x[-1], y[-1]), xytext=(8, dy.get(key, 0)), textcoords="offset points",
                    va="center", color=INK, fontsize=9)
    ax.set_xscale("log", base=2)
    ax.set_xticks(x); ax.set_xticklabels([str(v) for v in x]); ax.minorticks_off()
    ax.set_xlabel(xlabel); ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlim(x[0] / 1.3, x[-1] * 3.2)


draw(axes[0], fr, {"flat": ("flat", "flat ranking"), "graph": ("graph_bfs_s3", "graph BFS (3 seeds)"),
                   "hybrid": ("hybrid_half", "half flat + 1-hop")},
     "(a) Finding: any gold node read", "symbols read (budget)", dy={"flat": 5, "hybrid": -5})
draw(axes[1], co, {"flat": ("flat", "flat ranking"), "graph": ("graph:all", "graph BFS, all edges"),
                   "hybrid": ("hybrid:all", "alternate graph/flat"), "calls": ("graph:calls", "graph BFS, calls only")},
     "(b) Completing: remaining gold read, given one", "additional symbols read (budget)")
axes[0].set_ylabel("fraction of tasks / gold recovered"); axes[0].set_ylim(0, 1)
fig.tight_layout()
fig.savefig("paper/fig_find_vs_complete.png", dpi=200)
print("saved")
