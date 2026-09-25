#!/usr/bin/env python3

import os
import math
import json
import logging
import traceback
from datetime import date, datetime, timezone, timedelta
from flask import Flask, render_template, request, jsonify, url_for
from markupsafe import Markup
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
# RADAR CHART → INLINE SVG (no charting library, no image)
# =====================================================
# Only 3 axes, so the geometry is just a triangle — plain trig, no need for
# matplotlib. Building it as raw <svg> markup means it's part of the page's
# own DOM, so it can use the site's real CSS variables and fonts (var(--…))
# directly instead of being a separately-styled, rasterized PNG.
RADAR_SIZE = 380
RADAR_CENTER = RADAR_SIZE / 2
RADAR_OUTER_R = 100
RADAR_INNER_OFFSET = 0.15   # inner-triangle baseline, so a 0 on every axis
                            # still traces a visible (small) triangle
RADAR_LABEL_R = RADAR_OUTER_R * 1.42


def _radar_theta(axis_index, n=3):
    return math.pi / 2 - axis_index * (2 * math.pi / n)


def _radar_point(axis_index, frac, n=3):
    """frac in [0, 1]: 0 = inner-offset boundary, 1 = outer radius."""
    theta = _radar_theta(axis_index, n)
    inner_r = RADAR_OUTER_R * RADAR_INNER_OFFSET
    r = inner_r + frac * (RADAR_OUTER_R - inner_r)
    x = RADAR_CENTER + r * math.cos(theta)
    y = RADAR_CENTER - r * math.sin(theta)
    return x, y


def _wrap_label(text, max_len=13):
    """Break a long axis label onto a couple of lines so it doesn't run
    past the edge of the chart, without needing a text-layout library."""
    words = text.split()
    lines, current = [], ""
    for w in words:
        trial = f"{current} {w}".strip()
        if len(trial) > max_len and current:
            lines.append(current)
            current = w
        else:
            current = trial
    if current:
        lines.append(current)
    return lines or [text]


def _points_attr(points):
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)


def build_radar_svg(result):
    """Build the 3-axis (recency / issues / popularity) radar as a small,
    self-contained inline SVG string. Geometry mirrors the old matplotlib
    version exactly: each axis is normalized against its own worst-case
    anchor (0 to RECENCY_DAYS_ANCHORS[-1], etc.), clamped to [0, 1], then
    plotted on top of the inner-triangle baseline above."""
    n = 3
    labels = ["Days Since Last Commit", result["issue_metric_label"], "Popularity"]
    raw_vals = [
        result["days_since_last_commit"],
        result["issue_metric_raw"],
        result["popularity"],
    ]
    max_vals = [
        float(RECENCY_DAYS_ANCHORS[-1]),
        float(ISSUE_DAYS_ANCHORS[-1]),
        float(POPULARITY_LOG10_ANCHORS[-1]),
    ]
    fracs = [
        max(0.0, min(1.0, raw / mv)) if mv else 0.0
        for raw, mv in zip(raw_vals, max_vals)
    ]

    outer_pts = [_radar_point(i, 1.0, n) for i in range(n)]
    inner_pts = [_radar_point(i, 0.0, n) for i in range(n)]
    mid_pts = [_radar_point(i, 0.5, n) for i in range(n)]
    threeq_pts = [_radar_point(i, 0.75, n) for i in range(n)]
    data_pts = [_radar_point(i, f, n) for i, f in enumerate(fracs)]

    axis_scores = [
        result["recency_score_1to5"],
        result["issue_score_1to5"],
        result["pop_score_1to5"],
    ]

    # Native <title> elements — a real browser tooltip on hover, no JS needed.
    axis_tooltips = [
        f"{labels[0]}: {raw_vals[0]:.0f} days",
        f"{labels[1]}: {raw_vals[1]:.0f} days",
        f"{labels[2]}: {raw_vals[2]:.2f} (log\u2081\u2080 scale)",
    ]

    spokes = "".join(
        f'<line x1="{ix:.1f}" y1="{iy:.1f}" x2="{ox:.1f}" y2="{oy:.1f}" class="radar-grid-line" />'
        for (ix, iy), (ox, oy) in zip(inner_pts, outer_pts)
    )

    # Small marker dot at each vertex 
    markers = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" class="radar-data-point" '
        f'data-tooltip="{tip.replace(chr(34), "&quot;")}" />'
        for (x, y), tip in zip(data_pts, axis_tooltips)
    )

    label_svgs = []
    for i in range(n):
        theta = _radar_theta(i, n)
        lx = RADAR_CENTER + RADAR_LABEL_R * math.cos(theta)
        ly = RADAR_CENTER - RADAR_LABEL_R * math.sin(theta)
        anchor = ["middle", "start", "end"][i]
        lines = _wrap_label(labels[i])
        score_line = f"\u2605 {axis_scores[i]:.1f}"
        total_lines = len(lines) + 1
        start_dy = -((total_lines - 1) * 0.6)

        tspan_parts = [
            f'<tspan x="{lx:.1f}" dy="{(start_dy if j == 0 else 1.2):.2f}em">{line}</tspan>'
            for j, line in enumerate(lines)
        ]
        tspan_parts.append(
            f'<tspan x="{lx:.1f}" dy="1.2em" class="radar-label-score">{score_line}</tspan>'
        )
        tspans = "".join(tspan_parts)

        label_svgs.append(
            f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="{anchor}" class="radar-label">{tspans}</text>'
        )

    svg = f'''<svg viewBox="0 0 {RADAR_SIZE} {RADAR_SIZE}" class="radar-svg" role="img" aria-label="Radar chart of recency, issue backlog, and popularity">
  <polygon points="{_points_attr(outer_pts)}" class="radar-ring radar-ring-outer" />
  <polygon points="{_points_attr(threeq_pts)}" class="radar-ring radar-ring-mid" />
  <polygon points="{_points_attr(mid_pts)}" class="radar-ring radar-ring-mid" />
  <polygon points="{_points_attr(inner_pts)}" class="radar-ring radar-ring-inner" />
  {spokes}
  <polygon points="{_points_attr(data_pts)}" class="radar-data" />
  {markers}
  {''.join(label_svgs)}
</svg>'''

    return Markup(svg)


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
                radar_svg = build_radar_svg(result)
                badge_url, markdown = build_badge(result["final_score_1to5"])
                save_to_history(result)

                # Flask/Werkzeug URL-encodes the repo query value automatically.
                share_url = url_for("index", repo=result["repo"], _external=True)

                return render_template(
                    "result.html",
                    result=result,
                    radar_svg=radar_svg,
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