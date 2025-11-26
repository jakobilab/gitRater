#!/usr/bin/env python3
"""
Parallel GitHub Quality Scanner (Optimized + Extended)
------------------------------------------------------
Collects lightweight public metadata for repository quality assessment:
- Maintenance activity (commits, issues, PRs)
- Testing / CI adoption
- Popularity (stars, forks, watchers)
- Repository freshness (time since last commit)
- Bug-related metrics (bug commits, bug issues)
- Issue resolution time (average and median)
- Languages, Topics, and Discussions (GraphQL additions)
- Skips repositories with ≤ 5 files total (to ignore trivial repos)
Uses:
- 1 GraphQL call per repo
- 3 lightweight REST calls (commits + bug issues + general issues)
- Exponential backoff for rate limits and transient errors
"""

import sys
import time
import os
import csv
import random
import requests
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ------------------------
# CONFIG
# ------------------------
MAX_THREADS = 6
OUTPUT_CSV = "bioinfo_repo_quality.csv"
GRAPHQL_URL = "https://api.github.com/graphql"
REST_URL = "https://api.github.com"

BUG_KEYWORDS = ["bug", "fix", "error", "issue", "defect", "patch", "repair"]
TEST_KEYWORDS = [
    "test", "tests", "spec", "pytest.ini", "conftest.py", "unittest",
    "testthat", "runit", "inst/tests", "makefile.test", "t/",
]

MAX_RETRIES = 5
BACKOFF_BASE = 1.5

CSV_FIELDS = [
    "repo", "is_archived", "is_fork", "created_at", "pushed_at",
    "repo_age_days", "days_since_last_commit",
    "commit_count", "bug_commits",
    "total_issues", "open_issues", "closed_issues",
    "bug_issues", "open_bug_issues", "closed_bug_issues", "avg_bug_days_open",
    "avg_issue_days_open", "median_issue_days_open",
    "pull_requests", "open_prs", "merged_prs",
    "languages", "topics", "discussion_count", "discussion_titles",
    "has_CI", "ci_files",
    "has_tests", "test_files",
    "stars", "forks", "watchers",
    "error"
]


# ------------------------
# SAFE REQUEST FUNCTION
# ------------------------
def safe_request(method, url, headers=None, params=None, json=None):
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.request(method, url, headers=headers, params=params, json=json, timeout=30)
            if resp.status_code in (502, 503):
                wait = BACKOFF_BASE ** attempt
                print(f"⚠️ Server error {resp.status_code}, retrying in {wait:.1f}s...")
                time.sleep(wait)
                continue
            if resp.status_code == 403 and "rate limit" in resp.text.lower():
                print("⚠️ Rate limit hit, sleeping 5 minutes...")
                time.sleep(300)
                continue
            resp.raise_for_status()
            return resp
        except (requests.Timeout, requests.ConnectionError) as e:
            wait = BACKOFF_BASE ** attempt
            print(f"⏳ Network error ({e}), retrying in {wait:.1f}s...")
            time.sleep(wait)
        except requests.RequestException as e:
            print(f"❌ Permanent failure: {e}")
            break
    return None


# ------------------------
# REPO SCAN
# ------------------------
def scan_repo(token, repo_full_name):
    owner, name = repo_full_name.split("/", 1)
    headers = {"Authorization": f"Bearer {token}"}

    result = {f: "" for f in CSV_FIELDS}
    result["repo"] = repo_full_name

    try:
        # ------------------------
        # 1️⃣ GRAPHQL CALL
        # ------------------------
        gql_resp = safe_request(
            "POST", GRAPHQL_URL, headers=headers,
            json={"query": GRAPHQL_QUERY, "variables": {"owner": owner, "name": name}},
        )
        if not gql_resp:
            result["error"] = "GraphQL request failed"
            return result

        data = gql_resp.json()
        if "errors" in data:
            result["error"] = str(data["errors"])
            return result

        repo_data = data.get("data", {}).get("repository")
        if not repo_data:
            result["error"] = "Repository not found or inaccessible"
            return result

        # Basic metadata
        result["is_archived"] = repo_data.get("isArchived", False)
        result["is_fork"] = repo_data.get("isFork", False)
        result["created_at"] = repo_data.get("createdAt", "")
        result["pushed_at"] = repo_data.get("pushedAt", "")
        if result["created_at"]:
            created = datetime.fromisoformat(result["created_at"].replace("Z", "+00:00"))
            result["repo_age_days"] = (datetime.now(timezone.utc) - created).days
        result["stars"] = repo_data.get("stargazerCount", 0)
        result["forks"] = repo_data.get("forkCount", 0)
        result["watchers"] = repo_data.get("watchers", {}).get("totalCount", 0)

        # Languages
        langs = repo_data.get("languages", {}).get("edges", [])
        if langs:
            result["languages"] = ", ".join([f"{l['node']['name']}({l['size']})" for l in langs])

        # Topics
        topics = repo_data.get("repositoryTopics", {}).get("nodes", [])
        if topics:
            result["topics"] = ", ".join([t["topic"]["name"] for t in topics])

        # Discussions
        discussions = repo_data.get("discussions", {})
        result["discussion_count"] = discussions.get("totalCount", 0)
        if discussions.get("nodes"):
            result["discussion_titles"] = ", ".join([d["title"] for d in discussions["nodes"][:5]])

        # Issues & PRs
        result["total_issues"] = repo_data.get("issues", {}).get("totalCount", 0)
        result["open_issues"] = repo_data.get("open", {}).get("totalCount", 0)
        result["closed_issues"] = repo_data.get("closed", {}).get("totalCount", 0)
        result["pull_requests"] = repo_data.get("pullRequests", {}).get("totalCount", 0)
        result["open_prs"] = repo_data.get("openPRs", {}).get("totalCount", 0)
        result["merged_prs"] = repo_data.get("mergedPRs", {}).get("totalCount", 0)

        # Commits
        commit_target = repo_data.get("defaultBranchRef", {}).get("target", {})
        history = commit_target.get("history", {})
        recent_commit = (
            commit_target.get("recent", {}).get("nodes", [{}])[0]
            if commit_target.get("recent", {}).get("nodes")
            else {}
        )
        result["commit_count"] = history.get("totalCount", 0)
        last_commit_date = recent_commit.get("committedDate")
        if last_commit_date:
            last_commit_dt = datetime.fromisoformat(last_commit_date.replace("Z", "+00:00"))
            result["days_since_last_commit"] = (datetime.now(timezone.utc) - last_commit_dt).days

        # CI Detection
        ci_files = []
        for key, prefix in [
            ("githubDir", ".github"),
            ("workflowsDir", ".github/workflows"),
            ("circleDir", ".circleci"),
        ]:
            obj = repo_data.get(key)
            if obj and obj.get("entries"):
                ci_files += [f"{prefix}/{e['name']}" for e in obj["entries"]]
        if repo_data.get("travisFile"): ci_files.append(".travis.yml")
        if repo_data.get("appveyorFile"): ci_files.append("appveyor.yml")
        if repo_data.get("gitlabFile"): ci_files.append(".gitlab-ci.yml")
        result["has_CI"] = bool(ci_files)
        result["ci_files"] = ", ".join(ci_files)

        # ------------------------
        # 2️⃣ REST TREE SCAN — SKIP IF ≤ 5 FILES
        # ------------------------
        tree_url = f"{REST_URL}/repos/{owner}/{name}/git/trees/HEAD?recursive=1"
        tree_resp = safe_request("GET", tree_url, headers=headers)
        if not tree_resp or tree_resp.status_code != 200:
            result["error"] = "Failed to fetch tree"
            return result
        tree_data = tree_resp.json()
        paths = [t["path"].lower() for t in tree_data.get("tree", []) if t["type"] == "blob"]
        if len(paths) <= 5:
            print(f"⚠️ Skipping {repo_full_name} (only {len(paths)} files)")
            return None

        # Tests
        test_files = [p for p in paths if any(k in p for k in TEST_KEYWORDS)]
        result["has_tests"] = bool(test_files)
        result["test_files"] = ", ".join(test_files[:10])

        # ------------------------
        # 3️⃣ REST COMMITS (BUG DETECTION)
        # ------------------------
        commits_url = f"{REST_URL}/repos/{owner}/{name}/commits?per_page=100"
        commits_resp = safe_request("GET", commits_url, headers=headers)
        if commits_resp and commits_resp.status_code == 200:
            commits = commits_resp.json()
            bug_commits = [
                c for c in commits
                if any(k in c.get("commit", {}).get("message", "").lower() for k in BUG_KEYWORDS)
            ]
            result["bug_commits"] = len(bug_commits)
        else:
            result["bug_commits"] = 0

        # ------------------------
        # 4️⃣ REST BUG ISSUES
        # ------------------------
        issues_url = f"{REST_URL}/repos/{owner}/{name}/issues?state=all&labels=bug&per_page=100"
        issues_resp = safe_request("GET", issues_url, headers=headers)
        if issues_resp and issues_resp.status_code == 200:
            issues = [i for i in issues_resp.json() if "pull_request" not in i]
            result["bug_issues"] = len(issues)
            result["open_bug_issues"] = sum(1 for i in issues if i.get("state") == "open")
            result["closed_bug_issues"] = sum(1 for i in issues if i.get("state") == "closed")
            bug_durations = [
                (datetime.fromisoformat(i["closed_at"].replace("Z", "+00:00")) -
                 datetime.fromisoformat(i["created_at"].replace("Z", "+00:00"))).days
                for i in issues if i.get("created_at") and i.get("closed_at")
            ]
            result["avg_bug_days_open"] = round(sum(bug_durations) / len(bug_durations), 2) if bug_durations else ""
        else:
            result["bug_issues"] = result["open_bug_issues"] = result["closed_bug_issues"] = 0
            result["avg_bug_days_open"] = ""

        # ------------------------
        # 5️⃣ REST GENERAL ISSUES
        # ------------------------
        issues_url_all = f"{REST_URL}/repos/{owner}/{name}/issues?state=all&per_page=100"
        issues_resp_all = safe_request("GET", issues_url_all, headers=headers)
        if issues_resp_all and issues_resp_all.status_code == 200:
            all_issues = [i for i in issues_resp_all.json() if "pull_request" not in i]
            durations = [
                (datetime.fromisoformat(i["closed_at"].replace("Z", "+00:00")) -
                 datetime.fromisoformat(i["created_at"].replace("Z", "+00:00"))).days
                for i in all_issues if i.get("created_at") and i.get("closed_at")
            ]
            if durations:
                result["avg_issue_days_open"] = round(sum(durations) / len(durations), 2)
                result["median_issue_days_open"] = sorted(durations)[len(durations)//2]

    except Exception as e:
        print(f"💥 Unexpected error for {repo_full_name}: {e}")
        result["error"] = str(e)

    time.sleep(random.uniform(1.0, 2.0))
    return result


# ------------------------
# CSV WRITER
# ------------------------
def append_to_csv(row):
    if not row:
        return
    write_header = not os.path.exists(OUTPUT_CSV) or os.path.getsize(OUTPUT_CSV) == 0
    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ------------------------
# GRAPHQL QUERY (with new fields)
# ------------------------
GRAPHQL_QUERY = """
query($owner:String!, $name:String!) {
  repository(owner:$owner, name:$name) {
    nameWithOwner
    isPrivate
    isArchived
    isFork
    createdAt
    pushedAt
    stargazerCount
    forkCount
    watchers { totalCount }

    languages(first:10, orderBy:{field:SIZE, direction:DESC}) {
      edges { size node { name } }
      totalSize
    }

    repositoryTopics(first:10) { nodes { topic { name } } }

    discussions(first:10, orderBy:{field:CREATED_AT, direction:DESC}) {
      totalCount
      nodes { title createdAt url category { name } }
    }

    issues(states:[OPEN, CLOSED]) { totalCount }
    open: issues(states:OPEN) { totalCount }
    closed: issues(states:CLOSED) { totalCount }

    pullRequests(states:[OPEN, MERGED, CLOSED]) { totalCount }
    openPRs: pullRequests(states:OPEN) { totalCount }
    mergedPRs: pullRequests(states:MERGED) { totalCount }

    defaultBranchRef {
      name
      target {
        ... on Commit {
          history { totalCount }
          recent: history(first: 1) { nodes { committedDate } }
        }
      }
    }

    githubDir: object(expression: "HEAD:.github") { ... on Tree { entries { name type } } }
    workflowsDir: object(expression: "HEAD:.github/workflows") { ... on Tree { entries { name type } } }
    travisFile: object(expression: "HEAD:.travis.yml") { id }
    circleDir: object(expression: "HEAD:.circleci") { ... on Tree { entries { name } } }
    appveyorFile: object(expression: "HEAD:appveyor.yml") { id }
    gitlabFile: object(expression: "HEAD:.gitlab-ci.yml") { id }
  }
}
"""


# ------------------------
# MAIN
# ------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python check_repo_quality_extended.py repos.txt")
        sys.exit(1)

    repos_file = sys.argv[1]
    token = input("🔑 Enter your GitHub token: ").strip()

    with open(repos_file) as f:
        repos = [r.strip() for r in f if r.strip()]

    print(f"\n🚀 Starting scan of {len(repos)} repositories with {MAX_THREADS} threads...\n")

    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        future_to_repo = {executor.submit(scan_repo, token, repo): repo for repo in repos}
        for future in tqdm(as_completed(future_to_repo), total=len(repos), desc="Scanning Repositories", ncols=100):
            repo_name = future_to_repo[future]
            try:
                result = future.result()
                if result:
                    append_to_csv(result)
                    print(f"✅ Saved {repo_name}")
            except Exception as e:
                print(f"💥 Error saving {repo_name}: {e}")

    print(f"\n✅ All done! Results saved to {OUTPUT_CSV}\n")


if __name__ == "__main__":
    main()
