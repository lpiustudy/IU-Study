# Munich X Topic Analysis for IU Study Project

`munich_simple.py` collects German and English X posts matching a Munich-related
query, cleans the text, and identifies common topics using TF-IDF and
non-negative matrix factorization (NMF).

## Setup

Use an activated Python virtual environment with `tweepy`, `nltk`,
`scikit-learn`, and `python-dotenv` installed. Place your API token in a local
`.env` file next to the script:

```text
X_BEARER_TOKEN=your_token
```

Keep `.env` and collected data out of Git using `.gitignore`.

## 1. Collect

```bash
python munich_simple.py collect
```

Retrieves posts using the query, dates, and post limit configured in the script.
Set `ALLOW_API_REQUESTS = True` to enable paid collection. Blank dates use six
completed UTC days. The script waits at API rate limits and saves posts and
collection settings in `data/`. Existing `data/raw.json` is reused without new
requests; preserve the old data folder before starting a fresh collection.

## 2. Prepare

```bash
python munich_simple.py prepare
```

Removes duplicate text, URLs, mentions, punctuation, and language-specific
stopwords. Saves cleaned text in `data/prepared.csv` and counts hashtags and
author activity. NLTK stopwords are downloaded once if needed.

## 3. Analyze

```bash
python munich_simple.py analyze
```

Converts cleaned text into TF-IDF features and fits ten NMF topic components.
Assigns posts automatically and reports the five largest components using their
leading terms as labels, with counts and sample shares.

Results are saved in `outputs_auto/`. Open `report.md` for the topic ranking;
CSV files contain topic assignments, examples, hashtags, and author counts.
Model settings and diagnostics are recorded in `analysis.json`.

`prepare` and `analyze` do not make X API requests. No manual row labels are needed.
