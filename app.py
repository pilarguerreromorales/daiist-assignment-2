"""
Assignment 2 dashboard: hard-hat compliance from site camera snapshots.

Run with `uv run python main.py app`. Nothing is trained here: the app loads the committed
weights in weights/ and the evaluation train.ipynb saved in outputs/, then recomputes costs
live from the per-image test scores with the same formulas as the notebook (steps 6 and 7).

One case is one camera snapshot. Its score is the highest bare-head confidence in it.
Two thresholds split scores into pass (nobody looks), review (a person looks at the photo)
and fail (the safety officer is sent straight away).
"""
import json
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
FIGS = OUT / "figures"
EXAMPLES = ROOT / "assets" / "examples"

OP = json.loads((OUT / "operating_point.json").read_text())
BAND = OP["review_band"]
TEST = pd.read_csv(OUT / "test_image_scores.csv")
Y = TEST.has_head.to_numpy(bool)  # test snapshot contains at least one bare head
SCORES = {"finetuned": TEST.score_finetuned.to_numpy(), "coco_zero_shot": TEST.score_coco_zero_shot.to_numpy()}
PER_CLASS = pd.read_csv(OUT / "per_class_metrics.csv")
DATASET = pd.read_csv(OUT / "dataset_summary.csv", index_col=0)
MODEL = YOLO(ROOT / OP["weights"])

CHECK_MINUTES = BAND["check_minutes"]
DEFAULTS = {
    "fn": OP["cost"]["FN"], "check": OP["cost"]["FP"], "base": OP["base_rate"],
    "review_min": BAND["review_minutes"], "miss": BAND["reviewer_miss_rate"],
    "cameras": BAND["cameras"], "every": BAND["snapshot_every_min"], "hours": BAND["work_hours"],
}
NO_AUTO_FAIL = 1.01  # a high threshold above any score: nothing is failed without review
GRID = np.append(np.round(np.arange(0, 1.0001, 0.01), 2), NO_AUTO_FAIL)
OUTCOMES = ["pass", "review", "fail"]
OUTCOME_COLORS = {"pass": "#16a34a", "review": "#f59e0b", "fail": "#dc2626"}
MODEL_COLORS = {"finetuned": "#2563eb", "coco_zero_shot": "#94a3b8", "predict_nothing": "#cbd5e1"}


# --- cost model (same formulas as train.ipynb) ---------------------------------------------
def flagged(s, thr):
    """A snapshot reaches a threshold when its best head box does; no head box never does."""
    return (s > 0) & (s >= thr)


def share_curves(s):
    """Share of violations and of compliant snapshots at or above each grid threshold."""
    return (np.array([flagged(s[Y], t).mean() for t in GRID]),
            np.array([flagged(s[~Y], t).mean() for t in GRID]))


CURVES = {name: share_curves(s) for name, s in SCORES.items()}


def shares(s, lo, hi):
    """(pass, review, fail) shares among violations and among compliant snapshots."""
    out = []
    for mask in (Y, ~Y):
        f_lo, f_hi = flagged(s[mask], lo).mean(), flagged(s[mask], hi).mean()
        out.append((1 - f_lo, f_lo - f_hi, f_hi))
    return out


def review_cost(a):
    return a["check"] * a["review_min"] / CHECK_MINUTES


def per_case_cost(pos, neg, a):
    (pp, pr, pf), (_, nr, nf) = pos, neg
    reviewed_pos = review_cost(a) + (1 - a["miss"]) * a["check"] + a["miss"] * a["fn"]
    pos_cost = pp * a["fn"] + pr * reviewed_pos + pf * a["check"]
    neg_cost = nr * review_cost(a) + nf * a["check"]
    return a["base"] * pos_cost + (1 - a["base"]) * neg_cost


def cases_per_day(a):
    return a["cameras"] * 60 / a["every"] * a["hours"]


def workload(pos, neg, a):
    """Daily cost, misses and person-time for one way of running the site."""
    (pp, pr, pf), (_, nr, nf) = pos, neg
    n, base = cases_per_day(a), a["base"]
    reviews = n * (base * pr + (1 - base) * nr)
    walks = n * (base * ((1 - a["miss"]) * pr + pf) + (1 - base) * nf)
    minutes = reviews * a["review_min"] + walks * CHECK_MINUTES
    per_case = per_case_cost(pos, neg, a)
    return {"per_1000": 1000 * per_case, "per_day": n * per_case, "missed": n * base * (pp + a["miss"] * pr),
            "reviews": reviews, "walks": walks, "minutes": minutes, "people": minutes / (a["hours"] * 60),
            "review_share": base * pr + (1 - base) * nr}


def band_costs(a, model="finetuned"):
    """€ per 1,000 snapshots for every (low, high) pair on the grid; NaN where high < low."""
    F_pos, F_neg = CURVES[model]
    lo, hi = np.meshgrid(np.arange(len(GRID)), np.arange(len(GRID)), indexing="ij")
    pos = (1 - F_pos[lo], F_pos[lo] - F_pos[hi], F_pos[hi])
    neg = (1 - F_neg[lo], F_neg[lo] - F_neg[hi], F_neg[hi])
    return np.where(hi >= lo, 1000 * per_case_cost(pos, neg, a), np.nan)


def optimum(a):
    """Cheapest review band and cheapest single threshold (no review) for these assumptions."""
    cost = band_costs(a)
    i, j = np.unravel_index(np.nanargmin(cost), cost.shape)
    k = int(np.nanargmin(np.diag(cost)[:-1]))
    return {"low": float(GRID[i]), "high": float(GRID[j]), "band_cost": float(cost[i, j]),
            "single": float(GRID[k]), "single_cost": float(cost[k, k]), "grid": cost}


def designs(a, lo, hi):
    s, best = SCORES["finetuned"], optimum(a)
    rows = {
        "Do nothing": workload((1, 0, 0), (1, 0, 0), a),
        "Person reviews every snapshot": workload((0, 1, 0), (0, 1, 0), a),
        f"Model alone (fail ≥ {best['single']:.2f})": workload(*shares(s, best["single"], best["single"]), a),
        f"Model + optimal band [{best['low']:.2f}, {best['high']:.2f})":
            workload(*shares(s, best["low"], best["high"]), a),
        f"Your band [{lo:.2f}, {hi:.2f})": workload(*shares(s, lo, hi), a),
    }
    return pd.DataFrame(rows).T, best


# --- presentation helpers -------------------------------------------------------------------
def style(fig, title, height=380):
    fig.update_layout(
        title={"text": title, "x": 0.01, "font": {"size": 15}},
        template="plotly_white", height=height, margin={"l": 10, "r": 10, "t": 56, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, system-ui, sans-serif", "color": "#64748b", "size": 12},
        legend={"orientation": "h", "y": -0.18, "x": 0},
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.25)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.25)", zeroline=False)
    return fig


def card(label, value, note="", tone=""):
    return (f'<div class="kpi {tone}"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div><div class="kpi-note">{note}</div></div>')


def kpi_html(a, lo, hi):
    table, best = designs(a, lo, hi)
    mine, nothing = table.iloc[-1], table.loc["Do nothing"]
    review_all = table.loc["Person reviews every snapshot"]
    gap = mine.per_1000 - best["band_cost"]
    gap_note = "at the cost optimum" if gap < 0.5 else f"€{gap:,.0f} above the optimum"
    return '<div class="kpi-grid">' + "".join([
        card("Cost per 1,000 snapshots", f"€{mine.per_1000:,.0f}", gap_note, "accent"),
        card("Cost per day", f"€{mine.per_day:,.0f}",
             f"vs €{nothing.per_day:,.0f} doing nothing · €{review_all.per_day:,.0f} reviewing all"),
        card("Sent to review", f"{mine.review_share:.0%}", f"{mine.reviews:.0f} photos a day"),
        card("People needed", f"{mine.people:.2f}", f"{mine.minutes:.0f} person-min a day"),
        card("Violations missed", f"{mine.missed:.2f}/day", f"of {cases_per_day(a) * a['base']:.1f} a day"),
        card("Cost-optimal band", f"{best['low']:.2f} – {best['high']:.2f}",
             f"€{best['band_cost']:,.0f} per 1,000 for these assumptions"),
    ]) + "</div>"


def confusion_fig(a, lo, hi):
    s = SCORES["finetuned"]
    outcome = np.select([flagged(s, hi), flagged(s, lo)], ["fail", "review"], "pass")
    rows = [("≥1 bare head", Y), ("all helmeted", ~Y)]
    counts = np.array([[(outcome[m] == o).sum() for o in OUTCOMES] for _, m in rows])
    share = counts / counts.sum(1, keepdims=True)
    per_1000 = np.vstack([share[0] * 1000 * a["base"], share[1] * 1000 * (1 - a["base"])])
    text = [[f"<b>{c}</b> ({p:.0%})<br>{k:.1f} per 1,000" for c, p, k in zip(cr, pr, kr)]
            for cr, pr, kr in zip(counts, share, per_1000)]
    fig = go.Figure(go.Heatmap(z=share, x=[o.upper() for o in OUTCOMES], y=[r for r, _ in rows], text=text,
                               texttemplate="%{text}", colorscale=[[0, "#f1f5f9"], [1, "#1e3a8a"]],
                               showscale=False, zmin=0, zmax=1, hoverinfo="skip"))
    fig.update_yaxes(autorange="reversed")
    return style(fig, f"Test split by outcome · per 1,000 at {a['base']:.0%} base rate", 300)


def cost_curve_fig(a, lo, hi):
    best = optimum(a)
    diag = {m: np.diag(band_costs(a, m))[:-1] for m in SCORES}
    x = GRID[:-1]
    fig = go.Figure()
    fig.add_vrect(x0=lo, x1=min(hi, 1), fillcolor=OUTCOME_COLORS["review"], opacity=0.12, line_width=0,
                  annotation_text="your review band", annotation_position="top left")
    for m, label in [("coco_zero_shot", "COCO zero-shot"), ("finetuned", "Fine-tuned (single threshold)")]:
        fig.add_scatter(x=x, y=diag[m], name=label, line={"color": MODEL_COLORS[m], "width": 3 if m == "finetuned" else 1.5})
    refs = {"Do nothing": 1000 * a["base"] * a["fn"],
            "Person reviews every snapshot": 1000 * per_case_cost((0, 1, 0), (0, 1, 0), a),
            "Send officer for every snapshot": 1000 * a["check"]}
    for (name, v), dash in zip(refs.items(), ["dash", "dot", "dashdot"]):
        fig.add_hline(y=v, line={"color": "#94a3b8", "dash": dash, "width": 1},
                      annotation_text=f"{name} · €{v:,.0f}", annotation_position="top right",
                      annotation_font_size=10)
    fig.add_scatter(x=[best["single"]], y=[best["single_cost"]], mode="markers", name=f"cheapest single threshold {best['single']:.2f}",
                    marker={"size": 12, "color": "#0f172a", "symbol": "circle"})
    fig.add_hline(y=best["band_cost"], line={"color": OUTCOME_COLORS["pass"], "width": 1.5},
                  annotation_text=f"optimal review band · €{best['band_cost']:,.0f}",
                  annotation_position="bottom right", annotation_font_size=10)
    top = max(refs.values()) * 1.08
    fig.update_yaxes(title="€ per 1,000 snapshots", range=[0, top])
    fig.update_xaxes(title="confidence threshold for 'bare head'", range=[0, 1])
    return style(fig, "Expected cost vs threshold", 420)


def designs_fig(a, lo, hi):
    table, _ = designs(a, lo, hi)
    colors = ["#cbd5e1", "#60a5fa", "#fb923c", "#16a34a", "#f59e0b"]
    fig = go.Figure(go.Bar(
        x=table.per_day, y=table.index, orientation="h", marker_color=colors,
        text=[f"€{c:,.0f}/day · {m:.0f} person-min · {k:.2f} missed" for c, m, k in
              zip(table.per_day, table.minutes, table.missed)],
        textposition="outside", cliponaxis=False, hoverinfo="skip"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="€ per day", range=[0, table.per_day.max() * 1.9])
    return style(fig, f"Ways to run the site · {cases_per_day(a):.0f} snapshots a day", 340)


def band_map_fig(a, lo, hi):
    best = optimum(a)
    cost = best["grid"]
    fig = go.Figure(go.Heatmap(z=cost, x=GRID, y=GRID, colorscale="Viridis", reversescale=True,
                               zmax=np.nanmin(cost) * 4, colorbar={"title": "€/1,000", "thickness": 10},
                               hovertemplate="low %{y:.2f} · high %{x:.2f}<br>€%{z:,.0f} per 1,000<extra></extra>"))
    fig.add_scatter(x=[best["high"]], y=[best["low"]], mode="markers", name="optimal",
                    marker={"size": 13, "color": "#dc2626", "line": {"color": "white", "width": 2}})
    fig.add_scatter(x=[hi], y=[lo], mode="markers", name="your band",
                    marker={"size": 13, "color": "#f59e0b", "symbol": "diamond", "line": {"color": "white", "width": 2}})
    fig.update_xaxes(title="high threshold (fail at or above)")
    fig.update_yaxes(title="low threshold (pass below)")
    return style(fig, "Every pair of thresholds", 420)


def miss_fig(a, lo, hi):
    rates = np.round(np.arange(0, 0.0501, 0.0025), 4)
    band, review_all = [], []
    for m in rates:
        b = {**a, "miss": m}
        band.append(cases_per_day(b) * np.nanmin(band_costs(b)) / 1000)
        review_all.append(cases_per_day(b) * per_case_cost((0, 1, 0), (0, 1, 0), b))
    fig = go.Figure()
    fig.add_scatter(x=rates * 100, y=review_all, name="Person reviews every snapshot", line={"color": "#60a5fa", "width": 2.5})
    fig.add_scatter(x=rates * 100, y=band, name="Model + optimal band", line={"color": "#16a34a", "width": 2.5})
    fig.add_vline(x=a["miss"] * 100, line={"dash": "dot", "color": "#0f172a"},
                  annotation_text=f"assumed {a['miss']:.1%}", annotation_position="top")
    fig.update_xaxes(title="reviewer misses this share of bare heads (%)")
    fig.update_yaxes(title="€ per day")
    return style(fig, "Is a human reviewer better than the model?", 420)


def base_rate_fig(a, lo, hi):
    rates = [0.01, 0.02, 0.05, 0.1, 0.2, 0.4]
    best = [optimum({**a, "base": r}) for r in rates]
    lows = [b["low"] for b in best]
    highs = [min(b["high"], 1) for b in best]
    singles = [b["single"] for b in best]
    x = [f"{r:.0%}" for r in rates]
    fig = go.Figure()
    fig.add_scatter(x=x, y=highs, name="high (fail at or above)", mode="lines+markers", line={"color": OUTCOME_COLORS["fail"]})
    fig.add_scatter(x=x, y=lows, name="low (pass below)", mode="lines+markers", line={"color": OUTCOME_COLORS["pass"]})
    fig.add_scatter(x=x, y=singles, name="single threshold (no review)", mode="lines+markers",
                    line={"color": "#0f172a", "dash": "dot"})
    fig.update_xaxes(title="share of snapshots with a bare head (base rate)")
    fig.update_yaxes(title="cost-optimal threshold", range=[-0.03, 1.03])
    return style(fig, "How the optimal thresholds move with the base rate", 420)


# --- model and dataset views (static: from the evaluation saved by train.ipynb) -------------
def per_class_table():
    t = PER_CLASS.copy()
    t["model"] = t.model.map({"finetuned": "Fine-tuned", "coco_zero_shot": "COCO zero-shot",
                              "predict_nothing": "Predict nothing"})
    t = t.rename(columns={"cls": "class", "precision": f"precision @{OP['det_conf']}",
                          "recall": f"recall @{OP['det_conf']}"})
    return t.round(3).fillna("—")


def ap_fig():
    fig = go.Figure()
    for cls, color in [("head", "#dc2626"), ("helmet", "#16a34a")]:
        d = PER_CLASS[PER_CLASS.cls == cls]
        fig.add_bar(x=d.model.str.replace("_", " "), y=d.AP50, name=f"{cls} AP50", marker_color=color,
                    text=d.AP50.round(3), textposition="outside")
    fig.update_yaxes(range=[0, 1.12], title="AP50")
    return style(fig, "AP50 per class on the test split", 360)


def detection_cm_fig(model):
    cm = pd.DataFrame(OP["detection_confusion"][model])  # columns = predicted, index = truth
    labels = ["head", "helmet", "background"]
    cm = cm.loc[labels, labels]
    shown = cm.div(cm.sum(axis=1).replace(0, 1), axis=0)
    shown.loc["background"] = 0
    text = [[("n/a" if r == c == "background" else
              f"<b>{cm.loc[r, c]}</b>" + ("" if r == "background" else f"<br>{shown.loc[r, c]:.0%}"))
             for c in labels] for r in labels]
    fig = go.Figure(go.Heatmap(z=shown.values, x=labels, y=labels, text=text, texttemplate="%{text}",
                               colorscale=[[0, "#f1f5f9"], [1, "#1e3a8a"]], showscale=False, zmin=0, zmax=1,
                               hoverinfo="skip"))
    fig.update_yaxes(autorange="reversed", title="true")
    fig.update_xaxes(title="predicted")
    name = "Fine-tuned" if model == "finetuned" else "COCO zero-shot (person → head)"
    return style(fig, f"{name} · boxes at conf ≥ {OP['det_conf']}", 340)


def threshold_pr_table():
    rows = []
    for label, classes in OP["per_class_at_threshold"].items():
        for cls, v in classes.items():
            rows.append({"cut-off": label, "class": cls, "precision": round(v["precision"], 3),
                         "recall": round(v["recall"], 3)})
    return pd.DataFrame(rows)


def figure_gallery(names):
    return [(str(FIGS / f"{n}.png"), caption) for n, caption in names if (FIGS / f"{n}.png").exists()]


# --- live inspection ------------------------------------------------------------------------
def predict(image):
    """Every box the model draws on an uploaded photo (kept so threshold moves are instant)."""
    if image is None:
        return None
    r = MODEL.predict(image, conf=0.001, imgsz=OP["imgsz"], verbose=False)[0]
    return [{"cls": r.names[c], "conf": s, "box": b} for b, c, s in
            zip(r.boxes.xyxy.tolist(), r.boxes.cls.int().tolist(), r.boxes.conf.tolist())]


def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def render(image, boxes, lo, hi):
    if image is None or boxes is None:
        return None, '<div class="verdict empty">Upload a site photo or pick an example below.</div>'
    hi = max(hi, lo)
    heads = [b for b in boxes if b["cls"] == OP["positive_class"]]
    score = max((b["conf"] for b in heads), default=0.0)
    outcome = "fail" if flagged(score, hi) else "review" if flagged(score, lo) else "pass"

    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    width = max(2, img.width // 250)
    f = font(max(12, img.width // 45))
    show_heads_from = max(lo, 0.05)  # fainter boxes still count towards the score but clutter the view
    for b in sorted(boxes, key=lambda b: b["conf"]):
        if b["cls"] == "helmet" and b["conf"] >= OP["det_conf"]:
            color = OUTCOME_COLORS["pass"]
        elif b["cls"] == "head" and b["conf"] >= show_heads_from:
            color = OUTCOME_COLORS["fail"] if b["conf"] >= hi else OUTCOME_COLORS["review"]
        else:
            continue
        x1, y1, x2, y2 = b["box"]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        label = f"{'bare head' if b['cls'] == 'head' else 'helmet'} {b['conf']:.2f}"
        tw, th = draw.textbbox((0, 0), label, font=f)[2:]
        draw.rectangle([x1, max(0, y1 - th - 4), x1 + tw + 6, max(th + 4, y1)], fill=color)
        draw.text((x1 + 3, max(0, y1 - th - 3)), label, fill="white", font=f)

    action = {"pass": "Nobody needs to look.", "review": "A person checks the photo on screen.",
              "fail": "The safety officer is sent straight away."}[outcome]
    rule = {"pass": f"score {score:.3f} < low {lo:.2f}", "review": f"low {lo:.2f} ≤ score {score:.3f} < high {hi:.2f}",
            "fail": f"score {score:.3f} ≥ high {hi:.2f}"}[outcome]
    n_heads = sum(b["conf"] >= OP["det_conf"] for b in heads)
    verdict = (f'<div class="verdict {outcome}"><div class="verdict-tag">{outcome.upper()}</div>'
               f'<div class="verdict-body"><div class="verdict-action">{action}</div>'
               f'<div class="verdict-rule">{rule} · {n_heads} bare head(s) at conf ≥ {OP["det_conf"]}</div>'
               f'<div class="verdict-rule">score = highest bare-head confidence in the photo</div></div></div>')
    return img, verdict


# --- layout ---------------------------------------------------------------------------------
THEME = gr.themes.Base(
    primary_hue="amber", secondary_hue="blue", neutral_hue="slate", radius_size="lg",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
)

CSS = """
.gradio-container {max-width: 1320px !important; margin: auto;}
.hero {background: linear-gradient(135deg, #0f172a 0%, #1e293b 55%, #78350f 140%); border-radius: 20px;
       padding: 28px 32px; color: #f8fafc;}
.hero h1 {margin: 0 0 6px; font-size: 28px; font-weight: 800; letter-spacing: -0.02em; color: #f8fafc;}
.hero p {margin: 0; color: #cbd5e1; font-size: 15px; max-width: 900px;}
.pills {margin-top: 14px; display: flex; flex-wrap: wrap; gap: 8px;}
.pill {background: rgba(248,250,252,0.08); border: 1px solid rgba(248,250,252,0.16); color: #e2e8f0;
       padding: 4px 12px; border-radius: 999px; font-size: 12px;}
.kpi-grid {display: grid; grid-template-columns: repeat(auto-fit, minmax(175px, 1fr)); gap: 12px;}
.kpi {border: 1px solid var(--border-color-primary); background: var(--block-background-fill);
      border-radius: 16px; padding: 14px 16px;}
.kpi.accent {border-color: #f59e0b; box-shadow: 0 0 0 3px rgba(245,158,11,0.12);}
.kpi-label {font-size: 11px; text-transform: uppercase; letter-spacing: 0.07em; color: var(--body-text-color-subdued);}
.kpi-value {font-size: 26px; font-weight: 750; margin: 4px 0 2px; color: var(--body-text-color); letter-spacing: -0.02em;}
.kpi-note {font-size: 12px; color: var(--body-text-color-subdued);}
.flow {display: grid; grid-template-columns: 1fr 1fr 1.6fr; gap: 12px; align-items: stretch;}
.step {border: 1px solid var(--border-color-primary); border-radius: 16px; padding: 14px 16px;
       background: var(--block-background-fill);}
.step b {display: block; font-size: 14px; margin-bottom: 4px; color: var(--body-text-color);}
.step span {font-size: 13px; color: var(--body-text-color-subdued);}
.branches {display: grid; gap: 8px;}
.branch {border-radius: 12px; padding: 8px 12px; font-size: 13px; color: var(--body-text-color);
         border-left: 5px solid; background: var(--block-background-fill); border-top: 1px solid var(--border-color-primary);
         border-right: 1px solid var(--border-color-primary); border-bottom: 1px solid var(--border-color-primary);}
.branch.pass {border-left-color: #16a34a;} .branch.review {border-left-color: #f59e0b;} .branch.fail {border-left-color: #dc2626;}
.verdict {display: flex; gap: 16px; align-items: center; border-radius: 18px; padding: 18px 20px;
          border: 1px solid var(--border-color-primary); background: var(--block-background-fill);}
.verdict.empty {color: var(--body-text-color-subdued);}
.verdict-tag {font-size: 26px; font-weight: 800; letter-spacing: 0.04em; color: white; border-radius: 14px;
              padding: 14px 20px; min-width: 130px; text-align: center;}
.verdict.pass .verdict-tag {background: #16a34a;} .verdict.review .verdict-tag {background: #f59e0b;}
.verdict.fail .verdict-tag {background: #dc2626;}
.verdict-action {font-size: 16px; font-weight: 650; color: var(--body-text-color);}
.verdict-rule {font-size: 13px; color: var(--body-text-color-subdued); margin-top: 2px;}
.note {font-size: 13px; color: var(--body-text-color-subdued);}
.opbar {border: 1px solid #f59e0b !important; border-radius: 16px !important;}
"""

HERO = f"""
<div class="hero">
  <h1>Hard-hat compliance · pass, review or fail</h1>
  <p>A site camera takes a snapshot every few minutes. A fine-tuned YOLO model scores how likely it is that
  someone is not wearing a helmet, and two thresholds decide whether the snapshot passes, goes to a person
  for review, or sends the safety officer straight away. Every number below is recomputed live from the
  model's scores on the {len(TEST)} held-out test photos.</p>
  <div class="pills">
    <span class="pill">Roboflow · {OP['dataset']['workspace']}/{OP['dataset']['project']} v{OP['dataset']['version']}</span>
    <span class="pill">YOLO11n fine-tuned · mAP50 {OP['overall']['finetuned']['mAP50']:.3f}</span>
    <span class="pill">Positive class · bare head</span>
    <span class="pill">Missed violation €{DEFAULTS['fn']:.0f} · check €{DEFAULTS['check']:.0f}</span>
  </div>
</div>
"""

FLOW = f"""
<div class="flow">
  <div class="step"><b>1 · Snapshot</b><span>{DEFAULTS['cameras']} camera, one photo every {DEFAULTS['every']} min,
    {DEFAULTS['hours']} h day: {cases_per_day(DEFAULTS):.0f} photos a day. About {DEFAULTS['base']:.0%} show a bare head (assumed).</span></div>
  <div class="step"><b>2 · Score</b><span>The model finds every head and helmet. The photo's score is the highest
    bare-head confidence in it.</span></div>
  <div class="branches">
    <div class="branch pass"><b>Pass</b> · score below <i>low</i>: nobody looks. A miss costs €{DEFAULTS['fn']:.0f}.</div>
    <div class="branch review"><b>Review</b> · between: a person looks at the photo for {DEFAULTS['review_min']} min
      (€{review_cost(DEFAULTS):.2f}) and misses {DEFAULTS['miss']:.0%} of violations.</div>
    <div class="branch fail"><b>Fail</b> · score at or above <i>high</i>: the officer is sent ({CHECK_MINUTES} min, €{DEFAULTS['check']:.0f}).</div>
  </div>
</div>
"""

with gr.Blocks(title="Hard-hat compliance dashboard", fill_width=True) as demo:
    gr.HTML(HERO)

    with gr.Group(elem_classes="opbar"):
        with gr.Row(equal_height=True):
            low = gr.Slider(0, 1, value=BAND["low"], step=0.01, label="Low threshold · pass below",
                            info="snapshots scoring below this pass without anyone looking", scale=3)
            high = gr.Slider(0, 1.01, value=BAND["high"], step=0.01, label="High threshold · fail at or above",
                             info="snapshots at or above this send the officer straight away (1.01 = never)", scale=3)
            snap = gr.Button("Snap to cost-optimal", variant="primary", scale=1)

    with gr.Tabs():
        with gr.Tab("Overview"):
            kpi_overview = gr.HTML()
            gr.HTML(FLOW)
            designs_plot = gr.Plot(show_label=False)

        with gr.Tab("Inspect a photo"):
            with gr.Row():
                with gr.Column(scale=1):
                    photo = gr.Image(type="pil", label="Site photo", sources=["upload", "clipboard"], height=380)
                    gr.Examples(examples=[[str(p)] for p in sorted(EXAMPLES.glob("*.jpg"))], inputs=photo,
                                label="Examples from the test split")
                with gr.Column(scale=1):
                    verdict = gr.HTML()
                    annotated = gr.Image(label="What the model sees", interactive=False, height=420)
                    gr.HTML('<div class="note">Red = bare head at or above <i>high</i>, amber = bare head in the review '
                            f'band (boxes below 0.05 are hidden but still count), green = helmet at conf ≥ {OP["det_conf"]}.</div>')
            boxes_state = gr.State()

        with gr.Tab("Decision lab"):
            with gr.Row():
                with gr.Column(scale=1, min_width=300):
                    gr.Markdown("#### Assumptions\nChange any of them: costs, workload and the optimal band update live. "
                                "Defaults are the ones defended in REPORT.md.")
                    fn = gr.Slider(10, 1000, value=DEFAULTS["fn"], step=10, label="Cost of a missed bare head (€)")
                    check = gr.Slider(0.5, 10, value=DEFAULTS["check"], step=0.5,
                                      label=f"Cost of sending the officer ({CHECK_MINUTES} min, €)")
                    base = gr.Slider(0.01, 0.4, value=DEFAULTS["base"], step=0.01, label="Base rate (share of snapshots with a bare head)")
                    review_min = gr.Slider(0.25, 5, value=DEFAULTS["review_min"], step=0.25, label="Minutes to review one photo")
                    miss = gr.Slider(0, 0.05, value=DEFAULTS["miss"], step=0.0025, label="Reviewer miss rate")
                    with gr.Accordion("Site size", open=False):
                        cameras = gr.Slider(1, 50, value=DEFAULTS["cameras"], step=1, label="Cameras")
                        every = gr.Slider(1, 30, value=DEFAULTS["every"], step=1, label="Snapshot every (min)")
                        hours = gr.Slider(4, 24, value=DEFAULTS["hours"], step=1, label="Working hours a day")
                    reset = gr.Button("Reset assumptions", size="sm")
                with gr.Column(scale=3):
                    kpi_lab = gr.HTML()
                    cm_plot = gr.Plot(show_label=False)
                    curve_plot = gr.Plot(show_label=False)
            with gr.Row():
                band_plot = gr.Plot(show_label=False)
                miss_plot = gr.Plot(show_label=False)
            with gr.Row():
                base_plot = gr.Plot(show_label=False)
                designs_lab_plot = gr.Plot(show_label=False)

        with gr.Tab("Model"):
            gr.Markdown(f"#### Per-class results on the test split\nPrecision and recall at the conventional "
                        f"confidence {OP['det_conf']}; AP over all confidences. Positive class: **bare head**.")
            gr.Dataframe(per_class_table(), interactive=False, wrap=True)
            with gr.Row():
                gr.Plot(ap_fig(), show_label=False)
                with gr.Column():
                    gr.Markdown("#### Precision and recall at the cost-chosen threshold")
                    gr.Dataframe(threshold_pr_table(), interactive=False)
            with gr.Row():
                gr.Plot(detection_cm_fig("finetuned"), show_label=False)
                gr.Plot(detection_cm_fig("coco_zero_shot"), show_label=False)
            gr.Markdown("#### Where it fails")
            gr.Gallery(figure_gallery([("failure_recall_by_size", "Head recall by size"),
                                       ("failure_missed_heads", "Most missed bare heads"),
                                       ("failure_false_heads", "Most confident false bare heads"),
                                       ("image_level_scores", "Image scores and ROC"),
                                       ("training_curves", "Training curves (Colab T4)")]),
                       columns=3, height="auto", object_fit="contain", show_label=False)

        with gr.Tab("Dataset"):
            gr.Markdown(f"#### {OP['dataset']['workspace']} / {OP['dataset']['project']} · version {OP['dataset']['version']}")
            gr.Dataframe(DATASET.reset_index().rename(columns={"index": "split"}).round(4), interactive=False, wrap=True)
            gr.Gallery(figure_gallery([("eda_class_balance", "Class balance and image composition"),
                                       ("eda_object_size", "Object size"),
                                       ("eda_resolution", "Image resolution"),
                                       ("eda_samples", "Test samples with ground truth")]),
                       columns=2, height="auto", object_fit="contain", show_label=False)

    assumptions = [fn, check, base, review_min, miss, cameras, every, hours]
    keys = ["fn", "check", "base", "review_min", "miss", "cameras", "every", "hours"]

    def refresh(lo, hi, *values):
        a = dict(zip(keys, values))
        hi = max(hi, lo)
        kpis = kpi_html(a, lo, hi)
        return (kpis, kpis, confusion_fig(a, lo, hi), cost_curve_fig(a, lo, hi), band_map_fig(a, lo, hi),
                miss_fig(a, lo, hi), base_rate_fig(a, lo, hi), designs_fig(a, lo, hi), designs_fig(a, lo, hi))

    def snap_to_optimal(*values):
        best = optimum(dict(zip(keys, values)))
        return best["low"], best["high"]

    views = [kpi_overview, kpi_lab, cm_plot, curve_plot, band_plot, miss_plot, base_plot, designs_plot, designs_lab_plot]
    for control in [low, high, *assumptions]:
        control.change(refresh, [low, high, *assumptions], views)
    demo.load(refresh, [low, high, *assumptions], views)
    snap.click(snap_to_optimal, assumptions, [low, high])  # moving the sliders triggers the refresh
    reset.click(lambda: [DEFAULTS[k] for k in keys], None, assumptions)

    photo.change(predict, photo, boxes_state).then(render, [photo, boxes_state, low, high], [annotated, verdict])
    for control in [low, high]:
        control.change(render, [photo, boxes_state, low, high], [annotated, verdict])
    demo.load(render, [photo, boxes_state, low, high], [annotated, verdict])

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS)
