"""Apply the pre-registered decision rules and draw the figures.

    python scripts/analyze.py            # reads results/runs/*.json
Writes results/summary.json, results/summary.md and figures/*.png.
Works on a partial grid: missing cells are reported as missing.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs"
FIG = ROOT / "figures"
TS = [1, 2, 4, 8, 16, 32]
BITS = [32.0, 4.0, 1.58]
B_EFF = {32.0: 16.0, 4.0: math.log2(15), 1.58: math.log2(3)}  # float arm runs bf16 matmuls
B_LABEL = {32.0: "float (bf16)", 4.0: "int4", 1.58: "ternary"}
THETA = "0.9"
SEQ = 256

# reference palette, light mode (first three slots validate all-pairs)
COL = {32.0: "#2a78d6", 4.0: "#eb6834", 1.58: "#1baf7a"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
ARM_STYLE = {"tied": dict(ls="-", marker="o"), "untied_w": dict(ls="--", marker="s"),
             "untied_c": dict(ls=":", marker="^")}
ARM_LABEL = {"tied": "tied (1 block x T passes)", "untied_w": "untied, same W (T thin blocks)",
             "untied_c": "untied, same compute (T full blocks)"}


def style():
    plt.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
        "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 6,
    })


def load():
    rows = []
    for p in sorted(RUNS.glob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        for m in r["models"]:
            rows.append({"task": r["task"], "arm": r["arm"], "d": r["d"], "T": r["T"],
                         "bits": float(m["bits"]), "seed": m["seed"], "params": m["params"],
                         "eval": m["eval"], "history": m["history"], "steps": r["steps"],
                         "job": p.stem, "train_seconds": r["train_seconds"]})
    return rows


def nstar(row, t_test=None, theta=THETA):
    t = str(t_test if t_test is not None else row["T"])
    return row["eval"][t]["n_star"][theta] if t in row["eval"] else None


def cells(rows, long=False):
    """(task, arm, d, bits, T) -> list of per-seed n* at T_test = T."""
    out = defaultdict(list)
    for r in rows:
        if ("_x" in r["job"]) != long:
            continue
        out[(r["task"], r["arm"], r["d"], r["bits"], r["T"])].append(nstar(r))
    return out


def med(v):
    return statistics.median(v) if v else None


def spearman(x, y):
    def rank(a):
        order = sorted(range(len(a)), key=lambda i: a[i])
        rk = [0.0] * len(a)
        i = 0
        while i < len(a):
            j = i
            while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
                j += 1
            for k in range(i, j + 1):
                rk[order[k]] = (i + j) / 2
            i = j + 1
        return rk
    rx, ry = rank(x), rank(y)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def verdicts(c, rows):
    out = {}
    params = {(r["task"], r["arm"], r["d"]): r["params"] for r in rows if r["T"] == 32}
    for task in ("s5", "ca110", "mod7"):
        ds = (32, 64) if task != "mod7" else (32,)
        bits = BITS if task != "mod7" else [32.0, 1.58]
        grow, sat, per_cell = 0, 0, {}
        lv = {}
        for d in ds:
            for b in bits:
                v = {T: med(c.get((task, "tied", d, b, T), [])) for T in (8, 16, 32)}
                if None in v.values():
                    per_cell[f"d{d}_b{b}"] = "missing"
                    continue
                censored = v[32] >= SEQ
                g = censored or v[32] >= 1.2 * v[16]
                s = (not censored) and v[32] < 1.1 * v[16] and v[32] < 1.2 * v[8]
                grow += g
                sat += s
                per_cell[f"d{d}_b{b}"] = {"n8": v[8], "n16": v[16], "n32": v[32],
                                          "grows": g, "saturates": s}
                lv[(d, b)] = v[32]
        res = {"cells": per_cell, "grow": grow, "saturate": sat, "n_cells": len(ds) * len(bits)}
        hi, lo = (max(ds), 32.0), (32, 1.58)
        if hi in lv and lo in lv:
            res["insensitive"] = lv[lo] >= 0.75 * lv[hi]
            res["ratio_hi_lo"] = (lv[hi] / lv[lo]) if lv[lo] else float("inf")
            wb = [params.get((task, "tied", d), 0) * B_EFF[b] for (d, b) in lv]
            res["spearman_logWb_n32"] = spearman([math.log(x) for x in wb], list(lv.values()))
            res["tracking"] = res["spearman_logWb_n32"] >= 0.6 and lv[hi] >= 2 * lv[lo]
            need = 4 if task != "mod7" else max(1, round(4 / 6 * res["n_cells"]))
            if grow >= need and res["insensitive"]:
                res["verdict"] = "H_compute supported"
            elif sat >= need and res["tracking"]:
                res["verdict"] = "H_ceiling supported"
            else:
                res["verdict"] = "mixed"
        else:
            res["verdict"] = "incomplete"
        out[task] = res
    return out


def secondary(rows, c):
    sec = {"test_time": [], "tied_vs_untied_w": [], "tied_vs_untied_c": [], "training_limited": []}
    for r in rows:
        if r["arm"] != "tied" or "_x" in r["job"]:
            continue
        T = r["T"]
        n_t = nstar(r)
        n_2t = nstar(r, 2 * T)
        n_4t = nstar(r, 4 * T)
        sec["test_time"].append({"task": r["task"], "d": r["d"], "bits": r["bits"], "seed": r["seed"],
                                 "T": T, "n_T": n_t, "n_2T": n_2t, "n_4T": n_4t,
                                 "n_half": nstar(r, T // 2) if T > 1 else None})
        h = r["history"]
        if len(h) >= 5:
            last = h[-1]["n_star_0.9"]
            k = max(0, int(round(0.8 * len(h))) - 1)
            at80 = h[k]["n_star_0.9"]
            sec["training_limited"].append({"task": r["task"], "d": r["d"], "bits": r["bits"],
                                            "seed": r["seed"], "T": T, "n_at_80pct": at80, "n_final": last,
                                            "flag": last >= 1.1 * at80 and last > at80})
    for (task, arm, d, b, T), v in c.items():
        if arm == "tied":
            continue
        tied = med(c.get((task, "tied", d, b, T), []))
        key = "tied_vs_untied_w" if arm == "untied_w" else "tied_vs_untied_c"
        sec[key].append({"task": task, "d": d, "bits": b, "T": T, "untied": med(v), "tied": tied})
    return sec


def regression(c, rows):
    params = {(r["task"], r["arm"], r["d"]): r["params"] for r in rows}
    out = {}
    for task in ("s5", "ca110", "mod7"):
        X, y = [], []
        for (tk, arm, d, b, T), v in c.items():
            n = med(v)
            if tk != task or arm != "tied" or n is None or n <= 0 or n >= SEQ:
                continue
            X.append([1.0, math.log2(params[(tk, arm, d)]), math.log2(B_EFF[b]), math.log2(T)])
            y.append(math.log2(n))
        if len(y) >= 6:
            coef, *_ = np.linalg.lstsq(np.array(X), np.array(y), rcond=None)
            out[task] = {"n": len(y), "intercept": coef[0], "log2_W": coef[1], "log2_b": coef[2],
                         "log2_T": coef[3]}
    return out


# ------------------------------------------------------------------ figures
def ylog(ax):
    ax.set_yscale("log", base=2)
    ax.set_ylim(0.7, SEQ * 1.45)
    ticks = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    ax.set_yticks(ticks)
    ax.set_yticklabels([str(t) for t in ticks])
    ax.axhline(SEQ, color=INK2, lw=0.8, ls="-", alpha=0.6)


def xlog(ax, ts):
    ax.set_xscale("log", base=2)
    ax.set_xticks(ts)
    ax.set_xticklabels([str(t) for t in ts])
    ax.minorticks_off()


def floor0(v):
    return [x if (x is not None and x > 0) else 0.85 for x in v]


def fig_nstar_vs_T(c, task):
    ds = (32, 64) if task != "mod7" else (32,)
    bits = BITS if task != "mod7" else [32.0, 1.58]
    fig, axes = plt.subplots(1, len(ds), figsize=(5.2 * len(ds), 4.2), squeeze=False, sharey=True)
    for ax, d in zip(axes[0], ds):
        ref = [T if task == "ca110" else T + 1 for T in TS] if task != "mod7" else None
        if ref:
            ax.plot(TS, ref, color=INK2, lw=1, ls="-", marker=None, alpha=0.7)
            ax.annotate("hand-built ternary block", (TS[2], ref[2]), xytext=(4, 6),
                        textcoords="offset points", color=INK2, fontsize=8)
        for arm in ("tied", "untied_w", "untied_c"):
            for b in bits:
                pts = [(T, med(c.get((task, arm, d, b, T), []))) for T in TS]
                pts = [(T, v) for T, v in pts if v is not None]
                if not pts:
                    continue
                ax.plot([p[0] for p in pts], floor0([p[1] for p in pts]), color=COL[b],
                        **ARM_STYLE[arm], markeredgecolor=SURF, markeredgewidth=1.2,
                        alpha=1.0 if arm == "tied" else 0.75)
        xlog(ax, TS)
        ylog(ax)
        ax.set_title(f"{task}, d = {d}", loc="left", fontsize=11)
        ax.set_xlabel("T (passes for tied; layers for untied)")
    axes[0][0].set_ylabel(f"n* (largest size solved at acc >= {THETA})")
    axes[0][0].text(1.02, SEQ * 1.08, "all 256 positions solved", color=INK2, fontsize=8)
    handles = [plt.Line2D([], [], color=COL[b], lw=2, label=B_LABEL[b]) for b in bits]
    handles += [plt.Line2D([], [], color=INK2, **ARM_STYLE[a], label=ARM_LABEL[a])
                for a in ("tied", "untied_w", "untied_c")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(FIG / f"nstar_vs_T_{task}.png", dpi=160)
    plt.close(fig)


def fig_test_time(rows, task):
    ds = (32, 64) if task != "mod7" else (32,)
    bits = [32.0, 1.58]
    fig, axes = plt.subplots(len(bits), len(ds), figsize=(5.0 * len(ds), 3.6 * len(bits)),
                             squeeze=False, sharex=True, sharey=True)
    ramp = {1: "#86b6ef", 2: "#5598e7", 4: "#3987e5", 8: "#2a78d6", 16: "#1c5cab", 32: "#0d366b"}
    tt = [1, 2, 4, 8, 16, 32, 64, 128]
    for i, b in enumerate(bits):
        for j, d in enumerate(ds):
            ax = axes[i][j]
            for T in TS:
                rs = [r for r in rows if r["task"] == task and r["arm"] == "tied" and r["d"] == d
                      and r["bits"] == b and r["T"] == T and "_x" not in r["job"]]
                if not rs:
                    continue
                ys = [med([nstar(r, t) for r in rs if nstar(r, t) is not None]) for t in tt]
                ax.plot(tt, floor0(ys), color=ramp[T], marker="o", markersize=4, lw=1.6,
                        label=f"trained at T={T}")
                k = tt.index(T)
                ax.plot([T], floor0([ys[k]]), marker="o", markersize=8, color=ramp[T],
                        markeredgecolor=INK, markeredgewidth=1.0)
            xlog(ax, tt)
            ylog(ax)
            ax.set_title(f"{task}, d = {d}, {B_LABEL[b]}", loc="left", fontsize=10)
            if i == len(bits) - 1:
                ax.set_xlabel("passes at test time")
            if j == 0:
                ax.set_ylabel("n*")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, fontsize=8.5, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Tied models run for more or fewer passes than they were trained with "
                 "(outlined dot = training depth)", fontsize=10, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(FIG / f"test_time_{task}.png", dpi=160)
    plt.close(fig)


def fig_heatmap(rows, task, d=64, b=32.0, seed=1):
    rs = {r["T"]: r for r in rows if r["task"] == task and r["arm"] == "tied" and r["d"] == d
          and r["bits"] == b and r["seed"] == seed and "_x" not in r["job"]}
    if not rs:
        return
    Ts = sorted(rs)
    mat = np.array([rs[T]["eval"][str(T)]["acc"] for T in Ts])
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue", ["#f0efec", "#cde2fb", "#86b6ef", "#3987e5",
                                                      "#1c5cab", "#0d366b"])
    fig, ax = plt.subplots(figsize=(8, 2.6))
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest",
                   extent=(0.5, mat.shape[1] + 0.5, len(Ts) - 0.5, -0.5))
    ax.set_yticks(range(len(Ts)))
    ax.set_yticklabels([str(T) for T in Ts])
    ax.set_ylabel("T (train = test)")
    ax.set_xlabel("problem size n (position)")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.set_label("accuracy")
    ax.set_title(f"{task}: per-position accuracy, tied d={d}, {B_LABEL[b]}, seed {seed}",
                 loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / f"heatmap_{task}.png", dpi=160)
    plt.close(fig)


def fig_wb(c, rows):
    params = {(r["task"], r["arm"], r["d"]): r["params"] for r in rows}
    tasks = [t for t in ("s5", "ca110", "mod7") if any(k[0] == t for k in c)]
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.6 * len(tasks), 3.8), squeeze=False, sharey=True)
    for ax, task in zip(axes[0], tasks):
        for arm, mk in (("tied", "o"), ("untied_w", "s")):
            for (tk, a, d, b, T), v in c.items():
                if tk != task or a != arm or T != 32:
                    continue
                wb = params[(tk, a, d)] * B_EFF[b]
                ax.scatter([wb], floor0([med(v)]), color=COL[b], marker=mk, s=60 if d == 64 else 34,
                           edgecolor=SURF, linewidth=1.2, zorder=3)
        ax.set_xscale("log")
        ylog(ax)
        ax.set_xlabel("weight bits W*b")
        ax.set_title(f"{task}, T = 32", loc="left", fontsize=11)
    axes[0][0].set_ylabel("n*")
    handles = [plt.Line2D([], [], color=COL[b], marker="o", ls="", label=B_LABEL[b]) for b in BITS]
    handles += [plt.Line2D([], [], color=INK2, marker="o", ls="", label="tied"),
                plt.Line2D([], [], color=INK2, marker="s", ls="", label="untied, same W"),
                plt.Line2D([], [], color=INK2, marker="o", ls="", markersize=9, label="large marker: d = 64")]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(FIG / "nstar_vs_Wb.png", dpi=160)
    plt.close(fig)


def fig_training(rows):
    tasks = [t for t in ("s5", "ca110", "mod7") if any(r["task"] == t for r in rows)]
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.6 * len(tasks), 3.4), squeeze=False)
    ramp = {1: "#86b6ef", 2: "#5598e7", 4: "#3987e5", 8: "#2a78d6", 16: "#1c5cab", 32: "#0d366b"}
    for ax, task in zip(axes[0], tasks):
        for r in rows:
            if r["task"] != task or r["arm"] != "tied" or r["d"] != 32 or r["bits"] != 32.0 or "_x" in r["job"]:
                continue
            h = r["history"]
            ax.plot([x["step"] for x in h], floor0([x["n_star_0.9"] for x in h]), color=ramp[r["T"]],
                    lw=1.4, ls="-" if r["seed"] == 1 else "--",
                    label=f"T={r['T']}" if r["seed"] == 1 else None)
        ylog(ax)
        ax.set_xlabel("training step")
        ax.set_title(f"{task}: n* during training (tied, d=32, float)", loc="left", fontsize=9.5)
    axes[0][0].set_ylabel("n* (512 held-out sequences)")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, fontsize=8.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(FIG / "training_curves.png", dpi=160)
    plt.close(fig)


def table_md(c, task, arm, ds, bits):
    lines = [f"| {arm} | d | b | " + " | ".join(f"T={T}" for T in TS) + " |",
             "|---|---|---|" + "---|" * len(TS)]
    for d in ds:
        for b in bits:
            vals = []
            for T in TS:
                v = c.get((task, arm, d, b, T))
                vals.append("–" if not v else (f"{med(v):g}" + (f" ({'/'.join(str(x) for x in v)})"
                                                              if len(v) > 1 else "")))
            if all(x == "–" for x in vals):
                continue
            lines.append(f"| | {d} | {B_LABEL[b]} | " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main():
    style()
    FIG.mkdir(exist_ok=True)
    rows = load()
    if not rows:
        print("no results yet")
        return
    c = cells(rows)
    ver = verdicts(c, rows)
    sec = secondary(rows, c)
    reg = regression(c, rows)
    cl = cells(rows, long=True)
    summary = {"verdicts": ver, "regression": reg, "secondary": sec,
               "long_runs": {str(k): v for k, v in cl.items()},
               "cells": {str(k): v for k, v in c.items()}}
    (ROOT / "results" / "summary.json").write_text(json.dumps(summary, indent=1, default=float),
                                                    encoding="utf-8")
    md = ["# Results summary (generated by scripts/analyze.py)", "",
          f"n* at accuracy >= {THETA}; median over seeds (per-seed values in parentheses). "
          f"256 means every position solved (censored).", ""]
    for task in ("s5", "ca110", "mod7"):
        if not any(k[0] == task for k in c):
            continue
        ds = (32, 64) if task != "mod7" else (32,)
        bits = BITS if task != "mod7" else [32.0, 1.58]
        md += [f"## {task}", "", f"**Verdict: {ver[task]['verdict']}** "
               f"(cells growing {ver[task]['grow']}/{ver[task]['n_cells']}, "
               f"saturating {ver[task]['saturate']}/{ver[task]['n_cells']}"
               + (f", Wb-insensitive {ver[task].get('insensitive')}, Wb-tracking {ver[task].get('tracking')}, "
                  f"Spearman {ver[task].get('spearman_logWb_n32', float('nan')):.2f}" if 'insensitive' in ver[task] else "")
               + ")", ""]
        for arm in ("tied", "untied_w", "untied_c"):
            if any(k[0] == task and k[1] == arm for k in c):
                md += [table_md(c, task, arm, ds, bits), ""]
        if task in reg:
            g = reg[task]
            md += [f"Fit log2 n* = {g['intercept']:.2f} + {g['log2_W']:.2f} log2 W + {g['log2_b']:.2f} log2 b "
                   f"+ {g['log2_T']:.2f} log2 T over {g['n']} uncensored tied cells.", ""]
        tt = [x for x in sec["test_time"] if x["task"] == task and x["n_2T"] is not None]
        if tt:
            md += ["Test-time passes (tied, median over seeds and bit-widths): n*(2T)/n*(T) by training T:", ""]
            for T in TS:
                rs = [x for x in tt if x["T"] == T and x["n_T"]]
                if rs:
                    md.append(f"- T={T}: " + f"{med([x['n_2T'] / x['n_T'] for x in rs]):.2f}"
                              + f" (n=T: {med([x['n_T'] for x in rs]):g}, 2T: {med([x['n_2T'] for x in rs]):g}, "
                              f"4T: {med([x['n_4T'] for x in rs if x['n_4T'] is not None]):g})")
            md.append("")
        tl = [x for x in sec["training_limited"] if x["task"] == task]
        if tl:
            md += [f"Training-limited flags (n* rose >= 10% in the last 20% of steps): "
                   f"{sum(x['flag'] for x in tl)}/{len(tl)} tied models.", ""]
    long_lines = [f"- {k}: 3x steps n* = {v}, base n* = {c.get(k)}" for k, v in cl.items()]
    if long_lines:
        md += ["## 3x-steps check", ""] + long_lines + [""]
    (ROOT / "results" / "summary.md").write_text("\n".join(md), encoding="utf-8")
    for task in ("s5", "ca110", "mod7"):
        if any(k[0] == task for k in c):
            fig_nstar_vs_T(c, task)
            fig_test_time(rows, task)
            fig_heatmap(rows, task, d=64 if task != "mod7" else 32)
    fig_wb(c, rows)
    fig_training(rows)
    print("\n".join(md))


if __name__ == "__main__":
    main()
