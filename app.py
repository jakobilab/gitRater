#!/usr/bin/env python3
import os
import io
import base64
from flask import Flask, render_template, request
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import log1p
from flask import jsonify

from check_bioinfo_testing import scan_repo


# =====================================================
# CONFIG
# =====================================================
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")   # SAFER — set externally
BENCHMARK_FILE = "all_repos_with_scores.csv"

app = Flask(__name__)
 

# =====================================================
# LOAD BENCHMARK (USE R-COMPUTED VALUES)
# =====================================================
if not os.path.exists(BENCHMARK_FILE):
    raise FileNotFoundError("Missing all_repos_with_scores.csv")

bench = pd.read_csv(BENCHMARK_FILE)

# Use R-computed winsorized Z values directly
winsor_min = bench["z_winsor"].min()
winsor_max = bench["z_winsor"].max()

# For computing NEW repo Z-scores, we still need the raw distributions
mean_days = bench["days_since_last_commit"].mean()
sd_days   = bench["days_since_last_commit"].std()

mean_issue = bench["avg_issue_days_open"].mean()
sd_issue   = bench["avg_issue_days_open"].std()

mean_pop = bench["popularity"].mean()
sd_pop   = bench["popularity"].std()

# For radar chart scaling
global_max = bench[["days_since_last_commit", "avg_issue_days_open", "popularity"]].max()
global_min = bench[["days_since_last_commit", "avg_issue_days_open", "popularity"]].min()



# =====================================================
# CALCULATE REPO QUALITY — MATCHES R EXACTLY
# =====================================================
def get_repo_quality(repo_url, token):
    repo_full = repo_url.replace("https://github.com/", "").strip("/")
    result = scan_repo(token, repo_full)
    if not result:
        raise RuntimeError("Scan failed")

    # Popularity metric identical to R
    popularity = np.log1p(
        float(result.get("stars", 0)) +
        float(result.get("forks", 0)) +
        float(result.get("watchers", 0))
    )

    # Clean fields identical to your R pipeline
    days = result.get("days_since_last_commit")
    if days is None:
        days = bench["days_since_last_commit"].max()
    days = float(days)

    issues = result.get("avg_issue_days_open")
    if issues is None or issues == 0:
        issues = 1
    issues = float(issues)

    # --- Compute R-style Z-scores ---
    recency_z = -(days - mean_days) / sd_days
    issue_z   = -(issues - mean_issue) / sd_issue
    pop_z     =  (popularity - mean_pop) / sd_pop

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


    return {
        "repo": repo_full,
        "days_since_last_commit": days,
        "avg_issue_days_open": issues,
        "popularity": popularity,
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



# =====================================================
# RADAR CHART → BASE64 PNG
# =====================================================
def make_radar_chart(result):
    labels = ["Days Since Last Commit", "Avg Issue Days Open", "Popularity"]

    raw_vals = np.array([
        result["days_since_last_commit"],
        result["avg_issue_days_open"],
        result["popularity"]
    ])

    min_vals = np.array([
        global_min["days_since_last_commit"],
        global_min["avg_issue_days_open"],
        global_min["popularity"]
    ])
    max_vals = np.array([
        global_max["days_since_last_commit"],
        global_max["avg_issue_days_open"],
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
                    "issues_raw": result["avg_issue_days_open"],
                    "popularity_raw": round(result["popularity"], 2),

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

                # --- NEW VALUES SENT TO TEMPLATE ---
                badge_url=badge_url,
                markdown=markdown
            )

        except Exception as e:
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
            "markdown": markdown
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500



# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",   
        port=5000,        
        debug=True
    )

