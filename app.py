#!/usr/bin/env python3

import os
import io
import json
import base64
import logging
import traceback
from datetime import date, datetime, timezone, timedelta
from flask import Flask, render_template, request, jsonify, url_for
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from check_bioinfo_testing import scan_repo


GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")   # SAFER — set externally

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG)


class RepoNotEligibleError(Exception):
    """Raised when a repo exists and was reachable, but shouldn't be scored
    (private or archived) — only public, active repos are supported."""
    pass


# =====================================================
# LOCAL RATING HISTORY (public repos only — private/archived
# repos never reach save_to_history, since get_repo_quality
# raises before returning for those)
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_FILE = os.path.join(BASE_DIR, "rated_repos.json")
HISTORY_MAX_ENTRIES = 50   # how many ratings to keep on disk
HISTORY_DISPLAY_COUNT = 5  # how many to show on the index page


def load_history():
    try:
        with open(HISTORY_FILE, "r") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_to_history(result):
    """Append a completed, public-repo rating to the local JSON history file."""
    entry = {
        "repo": result["repo"],
        "rated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "recency_score": round(result["recency_score_1to5"], 2),
        "issue_score": round(result["issue_score_1to5"], 2),
        "pop_score": round(result["pop_score_1to5"], 2),
        "final_score": round(result["final_score_1to5"], 2),
        "activity_tag": result["activity_tag"],
        "issue_tag": result["issue_tag"],
        "popularity_tag": result["popularity_tag"],
    }

    history = load_history()
    history.append(entry)
    history = history[-HISTORY_MAX_ENTRIES:]

    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2)
    except OSError as e:
        app.logger.error("Failed to write history file: %s", e)

    return entry


def get_recent_history(n=HISTORY_DISPLAY_COUNT):
    """Return the n most recently rated *distinct* repos, newest first.
    The underlying JSON file still logs every run (so re-rating a repo
    isn't lost), but if a repo was rated more than once, only its most
    recent rating is shown here, so repeats don't crowd out other repos."""
    seen = set()
    deduped = []
    for entry in reversed(load_history()):
        repo = entry["repo"]
        if repo in seen:
            continue
        seen.add(repo)
        deduped.append(entry)
        if len(deduped) >= n:
            break
    return deduped


# =====================================================
# ABSOLUTE SCORING ANCHORS (tune these — no corpus needed)
# =====================================================


RECENCY_DAYS_ANCHORS  = [0,   14,  30,  60,  90,  180, 365, 730]
RECENCY_SCORE_ANCHORS = [5.0, 4.5, 4.0, 3.5, 3.0, 2.0, 1.3, 0.0]

# Median (or avg) days an issue stays open -> score. 
ISSUE_DAYS_ANCHORS  = [0,   7,   14,  30,  60,  120, 365]
ISSUE_SCORE_ANCHORS = [5.0, 4.5, 4.0, 3.5, 3.0, 2.0, 1.0]

# log10(stars + forks + watchers + 1) -> score. a point per order of magnitude
POPULARITY_LOG10_ANCHORS = [0.0, 1.0, 2.0, 3.0, 4.0]   
POPULARITY_SCORE_ANCHORS = [1.0, 2.0, 3.0, 4.0, 5.0]

# What to assume when GitHub gives us no data for a metric at all 

DAYS_IF_MISSING = RECENCY_DAYS_ANCHORS[-1]
ISSUE_DAYS_IF_MISSING = ISSUE_DAYS_ANCHORS[-1]


LANG_COLORS = [
    "#AB0520", "#1E5288", "#378DBD", "#E8A33D", "#1c7a34",
    "#6B4FA0", "#C2554D", "#4A7C59", "#8A8D91", "#D4A017",
]


def parse_languages(languages_str):
    """Parse the 'Name(size), Name2(size2)' string scan_repo returns into a
    list of {name, bytes, pct} dicts, sorted largest-first, with a color
    assigned from LANG_COLORS."""
    if not languages_str:
        return []

    langs = []
    for part in languages_str.split(","):
        part = part.strip()
        if not part or "(" not in part:
            continue
        name, _, rest = part.partition("(")
        size_str = rest.rstrip(")")
        try:
            size = int(size_str)
        except ValueError:
            continue
        langs.append({"name": name.strip(), "bytes": size})

    total = sum(l["bytes"] for l in langs)
    if total <= 0:
        return []

    langs.sort(key=lambda l: l["bytes"], reverse=True)
    for i, l in enumerate(langs):
        l["pct"] = round(100 * l["bytes"] / total, 1)
        l["color"] = LANG_COLORS[i % len(LANG_COLORS)]

    return langs


def _to_float_or_none(v):
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def score_recency(days_since_last_commit):
    return float(np.interp(days_since_last_commit, RECENCY_DAYS_ANCHORS, RECENCY_SCORE_ANCHORS))


def score_issue_backlog(issue_days_open):
    return float(np.interp(issue_days_open, ISSUE_DAYS_ANCHORS, ISSUE_SCORE_ANCHORS))


def score_popularity(total_reach):
    log_val = np.log10(1.0 + max(total_reach, 0.0))
    return float(np.interp(log_val, POPULARITY_LOG10_ANCHORS, POPULARITY_SCORE_ANCHORS))


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


def get_driver_summary(result):
    """Identify which component is pulling the final score up or down the
    most. All three sub-scores already live on the same 1-5 scale, so
    comparing them directly is meaningful — no z-scores needed."""
    components = [
        ("Recency (commit activity)", result["recency_score_1to5"]),
        ("Issue backlog health", result["issue_score_1to5"]),
        ("Popularity", result["pop_score_1to5"]),
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
        "strongest_score": round(strongest[1], 2),
        "weakest_label": weakest[0],
        "weakest_score": round(weakest[1], 2),
        "summary": summary,
    }


def get_repo_quality(repo_url, token):
    repo_full = repo_url.replace("https://github.com/", "").strip("/")
    result = scan_repo(token, repo_full)
    if not result:
        raise RuntimeError("Scan failed")

    # GitHub's GraphQL API returns `repository: null` both when a repo
    # truly doesn't exist AND when it exists but the token can't see it
    # (e.g. a private repo it has no access to) — there's no way to tell
    # those apart from the response, so scan_repo surfaces both as this
    # same "error" string. Left unchecked, every other field on `result`
    # stays at its default ("" / 0), so the app would otherwise silently
    # score a private/nonexistent repo using bogus all-zero data instead
    # of failing. Catch it here, before anything downstream trusts those
    # fields.
    if result.get("error"):
        err = str(result["error"])
        err_lower = err.lower()
        not_found_signals = ("not found", "inaccessible", "could not resolve")
        if any(signal in err_lower for signal in not_found_signals):
            raise RepoNotEligibleError(
                f"'{repo_full}' couldn't be accessed — it's either private or doesn't exist. "
                f"Only public repositories can be analyzed."
            )
        raise RuntimeError(err)

    if result.get("is_private"):
        raise RepoNotEligibleError(
            f"'{repo_full}' is a private repository. Only public repositories can be analyzed."
        )
    if result.get("is_archived"):
        raise RepoNotEligibleError(
            f"'{repo_full}' is archived. Only actively maintained public repositories can be analyzed."
        )

    stars = _to_float_or_none(result.get("stars")) or 0.0
    forks = _to_float_or_none(result.get("forks")) or 0.0
    watchers = _to_float_or_none(result.get("watchers")) or 0.0
    total_reach = stars + forks + watchers
    popularity_log = np.log10(1.0 + total_reach)

    days = _to_float_or_none(result.get("days_since_last_commit"))
    if days is None:
        days = float(DAYS_IF_MISSING)


    if _to_float_or_none(result.get("median_issue_days_open")) is not None:
        issue_raw = _to_float_or_none(result.get("median_issue_days_open"))
        issue_metric_label = "Median issue days open"
    elif _to_float_or_none(result.get("avg_issue_days_open")) is not None:
        issue_raw = _to_float_or_none(result.get("avg_issue_days_open"))
        issue_metric_label = "Avg issue days open"
    else:
        issue_raw = float(ISSUE_DAYS_IF_MISSING)
        issue_metric_label = "Issue days open (no data — treated as worst-case)"

    # Kept only as a friendly display stat, not used in scoring.
    backlog_health = 1.0 / (1.0 + issue_raw)

    recency_score_1to5 = score_recency(days)
    issue_score_1to5   = score_issue_backlog(issue_raw)
    pop_score_1to5      = score_popularity(total_reach)

    final_score = (recency_score_1to5 + issue_score_1to5 + pop_score_1to5) / 3.0

    last_commit_date, activity_tag = get_last_commit_date(days)
    issue_tag = get_score_tag(issue_score_1to5, ISSUE_HEALTH_TIERS)
    popularity_tag = get_score_tag(pop_score_1to5, POPULARITY_TIERS)

    result_out = {
        "repo": repo_full,
        "days_since_last_commit": days,
        "last_commit_date": last_commit_date.isoformat(),
        "activity_tag": activity_tag,
        "issue_metric_label": issue_metric_label,
        "issue_metric_raw": issue_raw,
        "issue_tag": issue_tag,
        "backlog_health": backlog_health,
        "popularity": popularity_log,
        "popularity_tag": popularity_tag,
        "final_score_1to5": final_score,
        "recency_score_1to5": recency_score_1to5,
        "issue_score_1to5": issue_score_1to5,
        "pop_score_1to5": pop_score_1to5,
        "languages": parse_languages(result.get("languages", "")),
        "topics": [t.strip() for t in result.get("topics", "").split(",") if t.strip()],
        "about": (result.get("about") or "").strip(),
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
        result["popularity"],
    ])

    min_vals = np.array([0.0, 0.0, 0.0])
    max_vals = np.array([
        float(RECENCY_DAYS_ANCHORS[-1]),
        float(ISSUE_DAYS_ANCHORS[-1]),
        float(POPULARITY_LOG10_ANCHORS[-1]),
    ])

    norm_vals = np.clip((raw_vals - min_vals) / (max_vals - min_vals + 1e-9), 0, 1)

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


def build_badge(score):
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
    return badge_url, markdown


# =====================================================
# ROUTES
# =====================================================
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        repo_url = request.form.get("repo_url")

        if repo_url:
            try:
                result = get_repo_quality(repo_url, GITHUB_TOKEN)
                chart = make_radar_chart(result)
                badge_url, markdown = build_badge(result["final_score_1to5"])
                save_to_history(result)

                # Flask/Werkzeug URL-encodes the repo query value automatically.
                share_url = url_for("index", repo=result["repo"], _external=True)

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

                        "recency_score": round(result["recency_score_1to5"], 2),
                        "issue_score": round(result["issue_score_1to5"], 2),
                        "pop_score": round(result["pop_score_1to5"], 2),
                    },
                    driver=result["driver"],
                    badge_url=badge_url,
                    markdown=markdown,
                    languages=result["languages"],
                    topics=result["topics"],
                    about=result["about"],
                    share_url=share_url,
                )

            except Exception as e:
                app.logger.error("Error in / (index) route:\n%s", traceback.format_exc())
                return render_template("index.html", error=str(e), history=get_recent_history())

        return render_template("index.html", history=get_recent_history())

    # GET. A shareable link like /?repo=owner%2Fname lands here. Rather than
    # running the (potentially multi-second) GitHub scan synchronously and
    # leaving the browser on a blank page while it waits, render the index
    # page immediately with the repo pre-filled and the loading overlay
    # already showing, then auto-submit the form — the existing POST branch
    # above (with its normal loading UX) does the actual analysis.
    auto_repo = request.args.get("repo")
    return render_template("index.html", history=get_recent_history(), auto_repo=auto_repo)


@app.route("/api/badge", methods=["GET"])
def api_badge():
    repo_url = request.args.get("repo")
    if not repo_url:
        return jsonify({"error": "Missing repo parameter ?repo="}), 400

    try:
        result = get_repo_quality(repo_url, GITHUB_TOKEN)
        badge_url, markdown = build_badge(result["final_score_1to5"])
        save_to_history(result)

        return jsonify({
            "repo": result["repo"],
            "score": result["final_score_1to5"],
            "badge_url": badge_url,
            "markdown": markdown,
            "last_commit_date": result["last_commit_date"],
            "activity_tag": result["activity_tag"],
            "driver": result["driver"],
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
        port=8030,
        debug=True
    )