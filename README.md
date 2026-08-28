# gitRater

A Flask app that scores any public GitHub repo's quality (1–5) based on commit
recency, issue backlog health, and popularity, then generates a shareable badge for it.

## Requirements
- Packages in `requirements.txt`
- A GitHub personal access token

## Run

```bash
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

export GITHUB_TOKEN=ghp_your_token_here

python app.py
```

