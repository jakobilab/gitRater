#!/usr/bin/env python3
import os
import io
import base64
import logging
import traceback
from datetime import date, timedelta
from flask import Flask, render_template, request
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import log1p
from flask import jsonify

from check_bioinfo_testing import scan_repo



GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")   # SAFER — set externally
BENCHMARK_FILE = "all_repos_with_scores.csv"

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG)



if not os.path.exists(BENCHMARK_FILE):
    raise FileNotFoundError("Missing all_repos_with_scores.csv")

bench = pd.read_csv(BENCHMARK_FILE)

# Use R-computed winsorized Z values directly
winsor_min = bench["z_winsor"].min()
winsor_max = bench["z_winsor"].max()

mean_days = bench["days_since_last_commit"].mean()
sd_days   = bench["days_since_last_commit"].std()

# R computes the issue component from median_issue_days_open, run through a

if "median_issue_days_open" in bench.columns:
    ISSUE_FIELD = "median_issue_days_open"
    ISSUE_FIELD_LABEL = "Median issue days open"
elif "avg_issue_days_open" in bench.columns:
    ISSUE_FIELD = "avg_issue_days_open"
    ISSUE_FIELD_LABEL = "Avg issue days open"
else:
    raise KeyError(
        "all_repos_with_scores.csv has neither 'median_issue_days_open' nor "
        "'avg_issue_days_open' — can't compute the issue-backlog component."
    )

# backlog_health = 1 / (1 + issue_days_open)  --  matches the R pipeline exactly
bench["backlog_health"] = 1.0 / (1.0 + bench[ISSUE_FIELD].astype(float))
mean_backlog = bench["backlog_health"].mean()
sd_backlog   = bench["backlog_health"].std()

mean_pop = bench["popularity"].mean()
sd_pop   = bench["popularity"].std()

# For radar chart scaling
global_max = bench[["days_since_last_commit", ISSUE_FIELD, "popularity"]].max()
global_min = bench[["days_since_last_commit", ISSUE_FIELD, "popularity"]].min()



def get_last_commit_date(days_since_last_commit):
    """Approximate calendar date of the last commit, plus a plain-language
    activity tag. GitHub only gives us a day count, so the date is an
    estimate relative to today."""
    approx_date = date.today() - timedelta(days=round(days_since_last_commit))

    if days_since_last_commit <= 30:
        tag = "Active"
    elif days_since_last_commit <= 90:
        tag = "Moderately active"
    elif days_since_last_commit <= 180:
        tag = "Slowing down"
    else:
        tag = "Stale"

    return approx_date, tag



ISSUE_HEALTH_TIERS = [(4.0, "Responsive"), (2.0, "Steady"), (0.0, "Backlogged")]
POPULARITY_TIERS   = [(4.0, "Popular"), (2.0, "Established"), (0.0, "Niche")]


def get_score_tag(score_1to5, tiers):
    """Map a 1-5 sub-score to a plain-language tier label."""
    for threshold, label in tiers:
        if score_1to5 >= threshold:
            return label
    return tiers[-1][1]


def _to_float_or_none(v):
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def get_driver_summary(result):
    """Identify which component is pulling the final score up or down the
    most. All three z-scores share the same units, so comparing them
    directly is meaningful."""
    components = [
        ("Recency (commit activity)", result["recency_z"]),
        ("Issue backlog health", result["issue_z"]),
        ("Popularity", result["pop_z"]),
    ]
    strongest = max(components, key=lambda c: c[1])
    weakest = min(components, key=lambda c: c[1])

    if strongest[0] == weakest[0]:
        summary = "All three components are roughly balanced for this repo."
    else:
        summary = (
            f"{strongest[0]} is pulling the score up the most. "
            f"{weakest[0]} is holding it back the most."
        )

    return {
        "strongest_label": strongest[0],
        "strongest_z": round(strongest[1], 3),
        "weakest_label": weakest[0],
        "weakest_z": round(weakest[1], 3),
        "summary": summary,
    }


def get_repo_quality(repo_url, token):
    repo_full = repo_url.replace("https://github.com/", "").strip("/")
    result = scan_repo(token, repo_full)
    if not result:
        raise RuntimeError("Scan failed")

    # Popularity metric identical to R
    stars = _to_float_or_none(result.get("stars")) or 0.0
    forks = _to_float_or_none(result.get("forks")) or 0.0
    watchers = _to_float_or_none(result.get("watchers")) or 0.0
    popularity = np.log1p(stars + forks + watchers)

    days = _to_float_or_none(result.get("days_since_last_commit"))
    if days is None:
        days = float(bench["days_since_last_commit"].max())

    issue_raw = _to_float_or_none(result.get(ISSUE_FIELD))
    if issue_raw is None:
        issue_raw = _to_float_or_none(result.get("median_issue_days_open"))
    if issue_raw is None:
        issue_raw = _to_float_or_none(result.get("avg_issue_days_open"))
    if issue_raw is None:
        issue_raw = float(bench[ISSUE_FIELD].max())

    # R's saturating transform: fast-closing issue trackers cluster near 1,
    # slow ones decay toward 0 without a hard linear penalty.
    backlog_health = 1.0 / (1.0 + issue_raw)

    # --- Z-scores (matches R exactly: recency_z = scale(-days),issue_z = scale(backlog_health), pop_z = scale(popularity)) ---
    recency_z = -(days - mean_days) / sd_days
    issue_z   = (backlog_health - mean_backlog) / sd_backlog
    pop_z     = (popularity - mean_pop) / sd_pop

    # Mean Z
    z_mean = np.mean([recency_z, issue_z, pop_z])

    # Winsorize using R's real winsor bounds
    z_w = np.clip(z_mean, winsor_min, winsor_max)

    # Final 1–5 R scaling
    final_score = 1 + (z_w - winsor_min) * (4 / (winsor_max - winsor_min))
    
    # 1. Winsorize individual Z-values to R bounds
    recency_w = np.clip(recency_z, winsor_min, winsor_max)
    issue_w   = np.clip(issue_z,   winsor_min, winsor_max)
    pop_w     = np.clip(pop_z,     winsor_min, winsor_max)

    # 2. Rescale to 1–5
    recency_score_1to5 = 1 + (recency_w - winsor_min) * (4 / (winsor_max - winsor_min))
    issue_score_1to5   = 1 + (issue_w   - winsor_min) * (4 / (winsor_max - winsor_min))
    pop_score_1to5     = 1 + (pop_w     - winsor_min) * (4 / (winsor_max - winsor_min))

    last_commit_date, activity_tag = get_last_commit_date(days)
    issue_tag = get_score_tag(issue_score_1to5, ISSUE_HEALTH_TIERS)
    popularity_tag = get_score_tag(pop_score_1to5, POPULARITY_TIERS)

    result_out = {
        "repo": repo_full,
        "days_since_last_commit": days,
        "last_commit_date": last_commit_date.isoformat(),
        "activity_tag": activity_tag,
        "issue_metric_field": ISSUE_FIELD,
        "issue_metric_label": ISSUE_FIELD_LABEL,
        "issue_metric_raw": issue_raw,
        "issue_tag": issue_tag,
        "backlog_health": backlog_health,
        "popularity": popularity,
        "popularity_tag": popularity_tag,
        "final_score_1to5": final_score,
        "recency_z": recency_z,
        "issue_z": issue_z,
        "pop_z": pop_z,
        "z_mean": z_mean,
        "z_winsor": z_w,
        "recency_score_1to5": recency_score_1to5,
        "issue_score_1to5": issue_score_1to5,
        "pop_score_1to5": pop_score_1to5
    }
    result_out["driver"] = get_driver_summary(result_out)
    return result_out



# =====================================================
# RADAR CHART → BASE64 PNG
# =====================================================
def make_radar_chart(result):
    labels = ["Days Since Last Commit", result["issue_metric_label"], "Popularity"]

    raw_vals = np.array([
        result["days_since_last_commit"],
        result["issue_metric_raw"],
        result["popularity"]
    ])

    min_vals = np.array([
        global_min["days_since_last_commit"],
        global_min[ISSUE_FIELD],
        global_min["popularity"]
    ])
    max_vals = np.array([
        global_max["days_since_last_commit"],
        global_max[ISSUE_FIELD],
        global_max["popularity"]
    ])

    norm_vals = (raw_vals - min_vals) / (max_vals - min_vals + 1e-9)

    inner_offset = 0.15
    shifted_vals = inner_offset + norm_vals * (1 - inner_offset)

    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])
    plot_vals = np.concatenate([shifted_vals, [shifted_vals[0]]])

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    inner_triangle = np.ones(n) * inner_offset
    inner_triangle = np.concatenate([inner_triangle, [inner_triangle[0]]])
    ax.plot(angles, inner_triangle, "--", color="gray", linewidth=1)

    outer_triangle = np.ones(n)
    outer_triangle = np.concatenate([outer_triangle, [outer_triangle[0]]])
    ax.plot(angles, outer_triangle, "--", color="gray", linewidth=1)

    for frac in [0.5, 0.75]:
        r = inner_offset + frac * (1 - inner_offset)
        tri = np.ones(n) * r
        tri = np.concatenate([tri, [tri[0]]])
        ax.plot(angles, tri, "--", color="gray", alpha=0.4)

    for i in range(n):
        ax.plot([angles[i], angles[i]], [inner_offset, 1], color="gray", alpha=0.5)

    ax.plot(angles, plot_vals, linewidth=2, color="tab:blue")
    ax.fill(angles, plot_vals, alpha=0.3, color="tab:blue")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels([])

    ax.set_title(
        f"Radar Chart (Inner Triangle Baseline)\n{result['repo']}",
        fontsize=16, fontweight="bold"
    )

    ax.text(
        0.5, -0.10,
        f"Quality Score: {result['final_score_1to5']:.2f} / 5",
        ha="center",
        transform=ax.transAxes,
        fontsize=14, fontweight="bold"
    )

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    buf.seek(0)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close(fig)

    return encoded



# =====================================================
# ROUTES
# =====================================================
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        repo_url = request.form.get("repo_url")

        try:
            # Compute score
            result = get_repo_quality(repo_url, GITHUB_TOKEN)
            chart = make_radar_chart(result)

            # --- NEW: Build badge URL using same logic as /api/badge ---
            score = result["final_score_1to5"]
            score_str = f"{score:.2f}"

            if score >= 4.0:
                color = "brightgreen"
            elif score >= 3.0:
                color = "green"
            elif score >= 2.0:
                color = "yellow"
            else:
                color = "red"

            badge_url = (
                f"https://img.shields.io/badge/"
                f"Repo%20Quality%20Score-{score_str}-{color}"
                f"?style=plastic"
            )

            markdown = f"![Repo Quality]({badge_url})"

            return render_template(
                "result.html",
                result=result,
                chart_data=chart,
                breakdown={
                    "days_raw": result["days_since_last_commit"],
                    "last_commit_date": result["last_commit_date"],
                    "activity_tag": result["activity_tag"],

                    "issue_label": result["issue_metric_label"],
                    "issues_raw": round(result["issue_metric_raw"], 2),
                    "backlog_health": round(result["backlog_health"], 3),
                    "issue_tag": result["issue_tag"],

                    "popularity_raw": round(result["popularity"], 2),
                    "popularity_tag": result["popularity_tag"],

                    "final_score": round(result["final_score_1to5"], 2),
                    "recency_z": round(result["recency_z"], 3),
                    "issue_z": round(result["issue_z"], 3),
                    "pop_z": round(result["pop_z"], 3),
                    "z_mean": round(result["z_mean"], 3),
                    "z_winsor": round(result["z_winsor"], 3),

                    "recency_score": round(result["recency_score_1to5"], 2),
                    "issue_score": round(result["issue_score_1to5"], 2),
                    "pop_score": round(result["pop_score_1to5"], 2),
                },
                driver=result["driver"],

                # --- NEW VALUES SENT TO TEMPLATE ---
                badge_url=badge_url,
                markdown=markdown
            )

        except Exception as e:
            app.logger.error("Error in / (index) route:\n%s", traceback.format_exc())
            return render_template("index.html", error=str(e))

    return render_template("index.html")




@app.route("/api/badge", methods=["GET"])
def api_badge():
    repo_url = request.args.get("repo")
    if not repo_url:
        return jsonify({"error": "Missing repo parameter ?repo="}), 400

    try:
        result = get_repo_quality(repo_url, GITHUB_TOKEN)

        score = result["final_score_1to5"]
        score_str = f"{score:.2f}"

        if score >= 4.0:
            color = "brightgreen"
        elif score >= 3.0:
            color = "green"
        elif score >= 2.0:
            color = "yellow"
        else:
            color = "red"

        badge_url = (
            f"https://img.shields.io/badge/"
            f"Repo Quality Score-{score_str}-{color}"
            f"?style=plastic"
        )

        markdown = f"![Repo Quality]({badge_url})"

        return jsonify({
            "repo": result["repo"],
            "score": score,
            "badge_url": badge_url,
            "markdown": markdown,
            "last_commit_date": result["last_commit_date"],
            "activity_tag": result["activity_tag"],
            "driver": result["driver"]
        })

    except Exception as e:
        app.logger.error("Error in /api/badge route:\n%s", traceback.format_exc())
        return jsonify({"error": str(e)}), 500



# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",   
        port= 8030,        
        debug=True
    )