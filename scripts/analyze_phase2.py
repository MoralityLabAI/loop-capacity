"""Phase-2 decision rules, tables and figures (docs/preregistration_phase2.md).

    python scripts/analyze_phase2.py
Writes results/phase2/summary.md, results/phase2/summary.json and
figures/phase2_*.png. Works on partial results.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
P2 = ROOT / "results" / "phase2"
FIG = ROOT / "figures"
BITS = [32.0, 4.0, 1.58]
B_EFF = {32.0: 16.0, 4.0: math.log2(15), 1.58: math.log2(3)}
B_LABEL = {32.0: "float (bf16)", 4.0: "int4", 1.58: "ternary"}
COL = {32.0: "#2a78d6", 4.0: "#eb6834", 1.58: "#1baf7a"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def style():
    plt.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
        "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 6,
    })


def med(v):
    v = [x for x in v if x is not None]
    return statistics.median(v) if v else None


# ------------------------------------------------------------------ Q1
def load_mem(task="memrand"):
    rows = []
    for p in sorted((P2 / "runs").glob(f"{task}_*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        T = r["T"]
        for m in r["models"]:
            ev = m["eval"][str(T)]
            bound = m["params"] * B_EFF[float(m["bits"])] + m["scale_bits"] + math.log2(max(T, 1))
            hist = [h["bits"] for h in m["history"] if "bits" in h]
            sat = len(hist) >= 2 and abs(hist[-1] - hist[-2]) <= 0.05 * max(abs(hist[-1]), 1.0)
            rows.append({"d": r["d"], "T": T, "bits": float(m["bits"]), "seed": m["seed"], "W": m["params"],
                         "stored": ev["bits"], "acc": ev["acc"], "bound": bound, "saturated": sat,
                         "history": hist})
    return rows


def q1(rows):
    out = {"cells": {}, "bound_violations": [], "verdicts": {}}
    for r in rows:
        if r["stored"] > r["bound"]:
            out["bound_violations"].append(r)
    for d in sorted({r["d"] for r in rows}):
        for b in BITS:
            cell = {T: med([r["stored"] for r in rows if r["d"] == d and r["bits"] == b and r["T"] == T])
                    for T in sorted({r["T"] for r in rows if r["d"] == d})}
            cell = {T: v for T, v in cell.items() if v is not None}
            out["cells"][f"d{d}_b{b}"] = cell
            if 4 in cell and len(cell) >= 2:
                tmax = max(cell)
                if cell[4] < 100:  # nothing stored at the reference T: the ratio is meaningless
                    ratio, verdict = float("nan"), "no signal"
                else:
                    ratio = cell[tmax] / cell[4]
                    verdict = "flat" if 0.75 <= ratio <= 1.25 else ("grows" if ratio >= 1.5 else "in between")
                out["verdicts"][f"d{d}_b{b}"] = {"T_max": tmax, "ratio_vs_T4": ratio, "verdict": verdict}
    return out


def fig_q1(rows, task="memrand"):
    ds = sorted({r["d"] for r in rows})
    if not ds:
        return
    fig, axes = plt.subplots(1, len(ds), figsize=(5.2 * len(ds), 4.0), squeeze=False, sharey=True)
    for ax, d in zip(axes[0], ds):
        Ts = sorted({r["T"] for r in rows if r["d"] == d})
        W = next(r["W"] for r in rows if r["d"] == d)
        for b in BITS:
            ys = [med([r["stored"] for r in rows if r["d"] == d and r["bits"] == b and r["T"] == T]) for T in Ts]
            pts = [(T, y / W) for T, y in zip(Ts, ys) if y is not None and y > 0]
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], color=COL[b], marker="o",
                        markeredgecolor=SURF, label=B_LABEL[b])
            ax.axhline(B_EFF[b], color=COL[b], lw=1, ls="--", alpha=0.8)
        ax.set_xscale("log", base=2)
        ax.set_xticks(Ts)
        ax.set_xticklabels([str(t) for t in Ts])
        ax.minorticks_off()
        ax.set_yscale("log", base=2)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xlabel("passes T")
        ax.set_title(f"{task}, tied d = {d} (W = {W:,})", loc="left", fontsize=11)
    axes[0][0].set_ylabel("stored bits per parameter")
    axes[0][0].text(1.0, 16 * 1.08, "dashed: claim A bound, b bits per weight", color=INK2, fontsize=8)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(FIG / f"phase2_memorization_{task}.png", dpi=160)
    plt.close(fig)


# ------------------------------------------------------------------ Q2
def load_extrap():
    out = {}
    for p in sorted(P2.glob("extrap_*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        out[p.stem] = r
    return out


def curve(models, passes, bits=None):
    ms = [m for m in models if bits is None or float(m["bits"]) == bits]
    return {t: med([m["eval"][str(t)]["n_star"]["0.9"] for m in ms if str(t) in m["eval"]]) for t in passes}


def load_ca64():
    out = {}
    for p in sorted((P2 / "runs").glob("ca110d64_*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        out[p.stem] = r
    return out


def q2(ex, ca):
    res = {}
    for name, r in ex.items():
        c = curve(r["models"], r["passes"])
        ok = 32 in c and 128 in c and c[32]
        res[name] = {"curve": c, "ratio_128_32": (c[128] / c[32]) if ok else None}
    for name, r in ca.items():
        passes = r["record"]
        c = curve(r["models"], passes)
        ok = 32 in c and 128 in c and c[32]
        res[name] = {"curve": c, "ratio_128_32": (c[128] / c[32]) if ok else None}
    verdicts = {}
    for name, v in res.items():
        rt = v["ratio_128_32"]
        if rt is None:
            continue
        local = "_w4" in name
        verdicts[name] = ("extrapolates" if rt >= 2 else "does not extrapolate") if local else \
            ("plateaus" if rt < 1.25 else "does not plateau")
    return res, verdicts


def fig_q2(ex, ca):
    panels = [("mod7, d = 64, 512-token sequences",
               {k: v for k, v in ex.items() if "mod7" in k and "d64_T32" in k}),
              ("ca110d64 (64 heads)", ca)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), squeeze=False)
    for ax, (title, group) in zip(axes[0], panels):
        for name, r in group.items():
            passes = r.get("passes") or r.get("record")
            local = "_w4" in name
            for b in BITS:
                c = curve(r["models"], passes, b)
                ts = [t for t in passes if c.get(t) is not None]
                if not ts:
                    continue
                ax.plot(ts, [max(c[t], 0.85) for t in ts], color=COL[b], ls="-" if local else "--",
                        marker="o" if local else "s", markersize=4, alpha=1.0 if local else 0.75)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xlabel("passes at test time (trained at 32)")
        ax.set_title(title, loc="left", fontsize=11)
        ax.axvline(32, color=INK2, lw=0.8, alpha=0.6)
    axes[0][0].set_ylabel("n*")
    handles = [plt.Line2D([], [], color=COL[b], lw=2, label=B_LABEL[b]) for b in BITS]
    handles += [plt.Line2D([], [], color=INK2, ls="-", marker="o", label="local attention (window 4)"),
                plt.Line2D([], [], color=INK2, ls="--", marker="s", label="global attention")]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8.5, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIG / "phase2_extrapolation.png", dpi=160)
    plt.close(fig)


def main():
    style()
    FIG.mkdir(exist_ok=True)
    P2.mkdir(parents=True, exist_ok=True)
    q1_all = {}
    mems = {t: load_mem(t) for t in ("memrand", "mem16")}
    ex, ca = load_extrap(), load_ca64()
    r2, v2 = q2(ex, ca)
    md = ["# Phase 2 summary (generated by scripts/analyze_phase2.py)", ""]
    for task, mem in mems.items():
        if not mem:
            continue
        r1 = q1(mem)
        q1_all[task] = r1
        note = " (original design, no signal; see amendment 1)" if task == "mem16" else ""
        md += [f"## Q1 ({task}{note}): memorization capacity against passes", "",
               "Stored bits (median over seeds); bound = W*b + scale bits + log2 T; "
               "bits/param in parentheses.", "",
               "| d | b | " + " | ".join(f"T={T}" for T in sorted({r['T'] for r in mem})) + " | bound | verdict |",
               "|---|---|" + "---|" * (len({r['T'] for r in mem}) + 2)]
        for d in sorted({r["d"] for r in mem}):
            for b in BITS:
                cell = r1["cells"].get(f"d{d}_b{b}", {})
                bound = med([r["bound"] for r in mem if r["d"] == d and r["bits"] == b])
                Wd = next(r["W"] for r in mem if r["d"] == d)
                vals = [f"{cell[T]:,.0f} ({cell[T] / Wd:.2f})" if T in cell else "–"
                        for T in sorted({r['T'] for r in mem})]
                v = r1["verdicts"].get(f"d{d}_b{b}", {})
                md.append(f"| {d} | {B_LABEL[b]} | " + " | ".join(vals) +
                          f" | {bound:,.0f} | {v.get('verdict', '–')} ({v.get('ratio_vs_T4', float('nan')):.2f}) |"
                          if bound else "")
        md += ["", f"Bound violations: {len(r1['bound_violations'])}.",
               f"Unsaturated models (last two logged values differ by > 5%): "
               f"{sum(not r['saturated'] for r in mem)} of {len(mem)}.", ""]
    if r2:
        md += ["## Q2: extrapolation in test-time passes", "",
               "| run | n* by test passes (median over bit-widths and seeds) | n*(128)/n*(32) | verdict |",
               "|---|---|---|---|"]
        for name, v in r2.items():
            c = ", ".join(f"{t}: {x:g}" for t, x in v["curve"].items() if x is not None)
            rt = v["ratio_128_32"]
            md.append(f"| {name} | {c} | {rt:.2f} | {v2.get(name, '–')} |" if rt is not None
                      else f"| {name} | {c} | – | – |")
        md.append("")
    (P2 / "summary.md").write_text("\n".join(md), encoding="utf-8")
    (P2 / "summary.json").write_text(json.dumps({"q1": q1_all, "q2": r2, "q2_verdicts": v2}, indent=1,
                                                default=str), encoding="utf-8")
    if mems.get("memrand"):
        fig_q1(mems["memrand"], "memrand")
    if ex or ca:
        fig_q2(ex, ca)
    print("\n".join(md))


if __name__ == "__main__":
    main()
