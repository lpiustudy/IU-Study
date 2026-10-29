"""Munich-related X discussions: collect -> prepare -> analyze, without manual coding."""

import csv  # Read/write tables; identifiers remain strings.
import html  # Decode HTML entities such as &amp;.
import json  # Save raw posts, collection settings and analysis settings.
import os  # Read the token from an environment variable.
import re  # Remove URLs/mentions and extract words.
import sys  # Select collect, prepare or analyze from the terminal.
import unicodedata  # Normalize equivalent character representations.
import warnings  # Record numerical convergence warnings in the analysis report.
from collections import Counter  # Count authors, hashtags and assignments.
from datetime import datetime, timedelta, timezone  # Validate dates and record UTC timestamps.
from importlib.metadata import version  # Record installed library versions for reproducibility.
from pathlib import Path  # Keep paths relative to this script.

# SETTINGS. Only collect can contact X; enable it explicitly when ready to pay.
ALLOW_API_REQUESTS = False
POST_LIMIT = 1000  # A returned-post cap, NOT a monetary spending limit.
START = "2026-10-28T00:00:00Z"  # Optional UTC timestamp. Blank uses six days before END.
END = "2026-10-29T00:00:00Z"  # Optional UTC timestamp. Blank uses today's 00:00 UTC.
QUERY = '(München OR Munich OR Muenchen OR #München OR #Munich) (lang:de OR lang:en) -is:retweet'
N_CANDIDATES = 10  # Discover more than five components, then rank by assigned posts.
RANDOM_SEED = 42
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUT = ROOT / "outputs_auto"  # Keep earlier manually reviewed outputs untouched.

# Remove query words and conversational filler, not substantive topic vocabulary.
EXTRA_STOPWORDS = {
    "münchen", "munich", "muenchen", "münchner", "münchener", "muc",
    "ja", "mal", "schon", "eigentlich", "halt", "eben", "gibt", "etwas",
    "einfach", "wirklich", "gerade", "immer", "mehr", "ne", "naja",
    "well", "really", "just", "like", "also", "get", "got", "would", "could",
    "time", "know", "come", "going", "saw", "year", "years", "day", "days",
    "next", "today", "people", "want", "said", "say", "think", "much", "even", "new",
    "heute", "morgen", "gestern", "leute", "sagen", "sagt", "macht", "machen",
}


def write_csv(path, rows, columns):
    """Write row dictionaries as UTF-8 CSV. Overwrite derived output, not raw input."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    """Read a local CSV as dictionaries; preserve IDs and other fields as strings."""
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def save_json(path, value):
    """Save readable JSON, converting datetime values to timestamp strings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def collect():
    """Fetch up to POST_LIMIT posts, following API pages and waiting at rate limits.

    Existing raw.json is reused without requests. Partial downloads are saved on
    failure, but not automatically resumed. Only this function can make X requests.
    """
    if (DATA / "raw.json").exists():
        print("Existing data/raw.json reused. No X request made; continue with prepare.")
        return
    if not ALLOW_API_REQUESTS:
        print("Collection disabled. Set ALLOW_API_REQUESTS=True only when ready to pay.")
        return
    now = datetime.now(timezone.utc)
    end = datetime.fromisoformat(END.replace("Z", "+00:00")) if END else now.replace(hour=0, minute=0, second=0, microsecond=0)
    start = datetime.fromisoformat(START.replace("Z", "+00:00")) if START else end - timedelta(days=6)
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("Dates need a timezone, such as Z for UTC.")
    if not now - timedelta(days=7) < start < end < now:
        raise ValueError("Recent search needs a completed interval within the previous seven days.")
    if not isinstance(POST_LIMIT, int) or POST_LIMIT < 10:
        raise ValueError("POST_LIMIT must be an integer of at least 10.")

    import tweepy  # Authenticate and request posts through X API v2.
    from dotenv import load_dotenv  # Load the local .env without printing credentials.

    load_dotenv(ROOT / ".env")
    client = tweepy.Client(bearer_token=os.environ["X_BEARER_TOKEN"], wait_on_rate_limit=True)
    posts, next_token = [], None
    complete = False
    settings = dict(query=QUERY, start=start.isoformat(), end=end.isoformat(), post_limit=POST_LIMIT)
    try:
        while POST_LIMIT - len(posts) >= 10:
            # Billable network call. No separate user-profile lookups are requested.
            response = client.search_recent_tweets(
                query=QUERY, start_time=start, end_time=end,
                max_results=min(100, POST_LIMIT - len(posts)), next_token=next_token,
                tweet_fields=["author_id", "created_at", "lang", "entities"],
            )
            posts.extend(dict(post.data) for post in response.data or [])
            if response.errors:
                raise RuntimeError("Partial X API error. Saved posts remain available; inspect before retrying.")
            next_token = (response.meta or {}).get("next_token")
            if not next_token:
                break
        complete = True
    finally:
        save_json(DATA / "raw.json", posts)
        save_json(DATA / "collection.json", {**settings, "received": len(posts),
                  "run_completed": complete, "more_results_available": bool(next_token),
                  "next_token": next_token, "retrieved_at": datetime.now(timezone.utc).isoformat()})
    print(f"Saved {len(posts)} posts. Continue with prepare; collection can be disabled again.")


def clean(text, stop_words):
    """Return lowercase alphabetic tokens without URLs, mentions or stopwords.

    Keep original text elsewhere. Hashtag words and umlauts survive. Numbers,
    emojis and punctuation are removed. No translation or sentiment inference.
    """
    text = unicodedata.normalize("NFKC", html.unescape(text)).casefold()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"(?<!\w)@\w+", " ", text)
    return " ".join(word for word in re.findall(r"[^\W\d_]+", text)
                    if len(word) > 1 and word not in stop_words)


def prepare():
    """Clean all supported-language posts automatically; no relevance questionnaire.

    Prefer raw.json. If absent, reuse an old full review.csv, ignoring its manual
    relevant/reason fields. Count entities before duplicate-text removal so each
    distinct post ID contributes once. Topic texts are deduplicated separately.
    """
    import nltk  # Download the stopword resource once if it is not yet installed.
    from nltk.corpus import stopwords  # German/English language-specific stopword lists.
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS  # Additional standard English function words.

    try:
        stops = {"de": set(stopwords.words("german")), "en": set(stopwords.words("english"))}
    except LookupError:
        if not nltk.download("stopwords", quiet=True):
            raise RuntimeError("Stopword download failed. Run: python -m nltk.downloader stopwords")
        stops = {"de": set(stopwords.words("german")), "en": set(stopwords.words("english"))}
    stops["en"].update(ENGLISH_STOP_WORDS)
    if (DATA / "raw.json").exists():
        raw = json.loads((DATA / "raw.json").read_text(encoding="utf-8"))
        source = "raw.json"
    else:
        raw = read_csv(DATA / "review.csv")
        source = "legacy review.csv (manual labels ignored; earlier collection exclusions still apply)"

    unique, seen_ids = [], set()
    for post in raw:
        post_id = str(post["id"])
        if post_id in seen_ids or post.get("lang") not in stops:
            continue
        seen_ids.add(post_id)
        tags = json.loads(post["hashtags"]) if "hashtags" in post else [
            tag["tag"] for tag in post.get("entities", {}).get("hashtags", [])]
        unique.append(dict(id=post_id, author_id=str(post.get("author_id", "")),
            lang=post["lang"], text=post["text"], hashtags=json.dumps([tag.casefold() for tag in tags], ensure_ascii=False)))
    if not unique:
        raise ValueError("No German/English posts found; check your source file.")
    authors = Counter(row["author_id"] for row in unique if row["author_id"])
    hashtags = Counter(tag for row in unique for tag in set(json.loads(row["hashtags"])))
    write_csv(OUTPUT / "authors.csv", [dict(author_id=a, posts=n, share=n/len(unique))
              for a, n in authors.most_common()], ["author_id", "posts", "share"])
    write_csv(OUTPUT / "hashtags.csv", [dict(hashtag=t, posts=n) for t, n in hashtags.most_common()], ["hashtag", "posts"])

    rows, seen_text = [], set()
    empty = duplicates = 0
    for row in unique:
        # Remove URL/mention variants for duplicate detection before stopwords.
        key = clean(row["text"], set())
        if key in seen_text:
            duplicates += 1
            continue
        seen_text.add(key)
        row["clean_text"] = clean(row["text"], stops[row["lang"]] | EXTRA_STOPWORDS)
        if not row["clean_text"]:
            empty += 1
            continue
        rows.append(row)
    write_csv(DATA / "prepared.csv", rows, ["id", "author_id", "lang", "text", "hashtags", "clean_text"])
    save_json(DATA / "preparation.json", dict(source=source, input_records=len(raw), entity_posts=len(unique),
              duplicate_ids_or_unsupported_language=len(raw)-len(unique), duplicate_texts=duplicates,
              empty_texts=empty, model_posts=len(rows), languages=dict(Counter(r["lang"] for r in rows)),
              extra_stopwords=sorted(EXTRA_STOPWORDS),
              effective_stopwords={lang: sorted(words | EXTRA_STOPWORDS) for lang, words in stops.items()}))
    print(f"Prepared {len(rows)} unique modelling texts; entity counts use {len(unique)} posts. No manual review needed.")


def analyze():
    """Fit pooled TF-IDF/NMF and export ranked terms, assignments, examples and a report.

    This is offline. Automatic labels are leading terms, not invented English
    summaries. German and English synonyms are not translated/aligned. Shares
    describe deduplicated modelling texts, not all X users or Munich residents.
    """
    import numpy as np  # Numeric arrays for topic scores, ranking and masks.
    from sklearn.feature_extraction.text import TfidfVectorizer  # Sparse post-by-term weights.
    from sklearn.decomposition import NMF  # Nonnegative document/topic and topic/term factors.
    from sklearn.exceptions import ConvergenceWarning  # Detect unfinished numerical optimization.

    rows = read_csv(DATA / "prepared.csv")
    if N_CANDIDATES < 5 or len(rows) < max(20, N_CANDIDATES * 2):
        raise ValueError("Need at least five candidates and max(20, 2*candidates) prepared posts; no topics fabricated.")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.85,
                                max_features=10000, sublinear_tf=True)
    matrix = vectorizer.fit_transform([row["clean_text"] for row in rows])
    if min(matrix.shape) < N_CANDIDATES:
        raise ValueError("Too few posts or recurring terms for the candidate count. Check the data/settings.")
    model = NMF(n_components=N_CANDIDATES, init="nndsvda", random_state=RANDOM_SEED, max_iter=1000)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        weights = model.fit_transform(matrix)
    # X ≈ W @ H. Balance the arbitrary component scales before selecting each post's maximum.
    scores = weights * np.linalg.norm(model.components_, axis=1)
    covered = scores.sum(axis=1) > 0
    assigned = scores.argmax(axis=1)
    names = vectorizer.get_feature_names_out()
    topics, examples = [], []
    for topic in range(N_CANDIDATES):
        indices = np.flatnonzero((assigned == topic) & covered)
        terms = names[model.components_[topic].argsort()[-10:][::-1]]
        topics.append(dict(topic_id=f"topic_{topic+1}", label=" / ".join(terms[:3]),
            top_terms=", ".join(terms), posts=len(indices), share=len(indices)/len(rows),
            de_posts=sum(rows[i]["lang"] == "de" for i in indices),
            en_posts=sum(rows[i]["lang"] == "en" for i in indices)))
        for i in sorted(indices, key=lambda i: scores[i, topic], reverse=True)[:3]:
            examples.append(dict(topic_id=f"topic_{topic+1}", id=rows[i]["id"], lang=rows[i]["lang"], text=rows[i]["text"]))
    topics.sort(key=lambda topic: (-topic["posts"], topic["topic_id"]))
    ranked = [topic for topic in topics if topic["posts"] > 0]
    for row, label, valid in zip(rows, assigned, covered):
        row["topic_id"] = f"topic_{label+1}" if valid else "unassigned"
    columns = ["topic_id", "label", "top_terms", "posts", "share", "de_posts", "en_posts"]
    write_csv(OUTPUT / "all_topics.csv", topics, columns)
    write_csv(OUTPUT / "top_five.csv", ranked[:5], columns)
    write_csv(OUTPUT / "post_topics.csv", rows, ["id", "author_id", "lang", "text", "clean_text", "topic_id"])
    write_csv(OUTPUT / "topic_examples.csv", examples, ["topic_id", "id", "lang", "text"])
    issues = ["Pooled TF-IDF does not align German/English synonyms; components can reflect language differences.",
              "A component can mix subjects. Automatic term labels and counts do not establish semantic coherence.",
              "Keywords do not establish residence or geographic relevance; other meanings of Munich may remain.",
              "Topic counts use deduplicated texts. Author/hashtag counts use unique post IDs before text deduplication."]
    if (~covered).sum():
        issues.append(f"{int((~covered).sum())} texts have zero topic loading and remain unassigned, included in share denominators.")
    if len(ranked) < 5:
        issues.append("Fewer than five components have assigned posts; no extra findings invented.")
    if len(ranked) > 5 and ranked[4]["posts"] == ranked[5]["posts"]:
        issues.append("Tie at fifth place: top_five.csv uses topic ID to break it; all_topics.csv preserves all tied counts.")
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        issues.append("NMF did not converge. Treat this result as provisional and investigate settings before reporting.")
    collection = json.loads((DATA / "collection.json").read_text(encoding="utf-8")) if (DATA / "collection.json").exists() else {}
    if not collection:
        issues.append("Original collection settings unavailable. Record the actual query and dates separately.")
    elif not collection.get("run_completed", True) or collection.get("more_results_available", collection.get("next_token")):
        issues.append("Collection was interrupted or capped with more results available; coverage is incomplete.")
    save_json(OUTPUT / "analysis.json", dict(model_posts=len(rows), assigned=int(covered.sum()),
        unassigned=int((~covered).sum()), candidate_components=N_CANDIDATES, seed=RANDOM_SEED,
        tfidf=dict(ngram_range=[1,2], min_df=2, max_df=0.85, max_features=10000, sublinear_tf=True),
        iterations=int(model.n_iter_), features=matrix.shape[1], collection=collection,
        versions={package:version(package) for package in ["scikit-learn", "nltk", "numpy"]}, warnings=issues))
    report = ["# Munich-related X discussion: automated exploratory results", "",
              f"Modelling texts: {len(rows)}. Assigned: {int(covered.sum())}. Shares use all modelling texts.", "",
              "| Topic terms (automatic label) | Posts | Share |", "|---|---:|---:|"]
    report += [f"| {t['label']} | {t['posts']} | {t['share']:.1%} |" for t in ranked[:5]]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "report.md").write_text("\n".join(report)+"\n", encoding="utf-8")
    print("Analysis complete: outputs_auto/report.md and top_five.csv. No manual labels or X requests required.")


if __name__ == "__main__":  # Importing this file does not start collection or analysis.
    stages = {"collect": collect, "prepare": prepare, "analyze": analyze}
    if len(sys.argv) != 2 or sys.argv[1] not in stages:
        raise SystemExit("Usage: python munich_simple.py collect|prepare|analyze")
    stages[sys.argv[1]]()
