"""Essential Munich topic-analysis workflow. Run one stage at a time."""

import csv  # Read and write CSV tables.
import html  # Convert HTML entities such as &amp; into ordinary characters.
import json  # Save raw API data and collection settings as JSON.
import os  # Read the bearer token from environment variables.
import re  # Find/remove text patterns such as URLs and mentions.
import sys  # Read the stage name supplied in the terminal.
import unicodedata  # Standardize equivalent Unicode characters.
from collections import Counter  # Count hashtags, authors, and topic labels.
from pathlib import Path  # Work with files and folders.

# SETTINGS: change these before live collection. Dates must be valid UTC timestamps.
START = "2026-10-28T00:00:00Z"  # Example format: 2026-10-22T00:00:00Z
END = "2026-10-29T00:00:00Z"  # End of your chosen interval, in the same format.
POST_LIMIT = 1000  # Maximum returned posts; this is NOT a monetary spending limit.
ALLOW_API_REQUESTS = False  # Change to True ONLY when ready for paid collection.
TOPICS_PER_LANGUAGE = 5  # Candidate components, not five final bilingual findings.
QUERY = (
    '(München OR Munich OR Muenchen OR #München OR #Munich) '
    '(lang:de OR lang:en) -is:retweet '
    '-"Bayern Munich" -"Bayern München" -"Bayern Munchen" '
    '-"Bayern Muenchen" -"Harry Kane" '
    '-#FCBayern -#BayernMunich -#DFBPokal'
)

# All files are relative to this script, so the terminal's location does not matter.
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUT = ROOT / "outputs"


def write_csv(path, rows, columns):
    """Write dictionaries to a CSV file, using the supplied column order.

    Creates parent folders. Overwrites an existing file at this path.
    Dictionary fields not listed in columns are ignored. Returns nothing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    """Return a list of row dictionaries from a CSV. Keep IDs/values as strings."""
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def collect():
    """Download posts once and save raw data/settings locally.

    Uses the SETTINGS above and X_BEARER_TOKEN from .env. This is the ONLY
    stage that can contact X. With ALLOW_API_REQUESTS=False, it only prints a
    message. Pagination fetches further pages until the cap or results end.
    Partial records are saved if a request fails; no automatic retries occur.
    """
    if not ALLOW_API_REQUESTS:
        print("No API request made. Set dates and explicitly enable collection first.")
        return
    if not START or not END:
        raise ValueError("Set START and END before collecting.")
    if not isinstance(POST_LIMIT, int) or POST_LIMIT < 10:
        raise ValueError("POST_LIMIT must be an integer of at least 10.")
    if (DATA / "raw.json").exists():
        raise ValueError("raw.json already exists. Reuse it instead of collecting again.")

    import tweepy  # X API client; imported only when live collection is enabled.
    from dotenv import load_dotenv  # Load local .env settings without hardcoding secrets.

    load_dotenv(ROOT / ".env")
    token = os.environ["X_BEARER_TOKEN"]
    client = tweepy.Client(bearer_token=token, wait_on_rate_limit=True)
    posts = []
    next_token = None  # API continuation token; None starts at the first page.
    try:
        while POST_LIMIT - len(posts) >= 10:
            # LIVE API REQUEST. Recent search only covers the previous seven days.
            # For an older period, change this method to search_all_tweets and
            # check your account access/pricing first. No user profiles are fetched.
            response = client.search_recent_tweets(
                query=QUERY, start_time=START, end_time=END,
                max_results=min(100, POST_LIMIT - len(posts)), next_token=next_token,
                tweet_fields=["author_id", "created_at", "lang", "entities"],
            )
            for post in response.data or []:
                posts.append(dict(post.data))
            if response.errors:
                raise RuntimeError("X returned partial errors. Check saved data before retrying.")
            next_token = (response.meta or {}).get("next_token")
            if not next_token:
                break
    finally:
        DATA.mkdir(parents=True, exist_ok=True)
        # default=str serializes any datetime values returned by Tweepy.
        (DATA / "raw.json").write_text(
            json.dumps(posts, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        (DATA / "collection.json").write_text(json.dumps({
            "query": QUERY, "start": START, "end": END, "post_limit": POST_LIMIT,
            "received": len(posts), "next_token": next_token,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(posts)} posts. You can disable collection again now.")


def prepare():
    """Turn raw posts into review.csv, removing duplicate IDs.

    Keeps original text and hashtag metadata. Adds relevant/reason columns for
    YOU to fill. Does not decide relevance or whether an author lives in Munich.
    Refuses to overwrite an existing review file. Returns nothing.
    """
    if (DATA / "review.csv").exists():
        raise ValueError("review.csv already exists; keep your existing review decisions.")
    posts = json.loads((DATA / "raw.json").read_text(encoding="utf-8"))
    rows = []
    seen = set()
    for post in posts:
        if str(post["id"]) in seen:
            continue
        seen.add(str(post["id"]))
        tags = [tag["tag"].casefold() for tag in post.get("entities", {}).get("hashtags", [])]
        rows.append({"id": str(post["id"]), "author_id": str(post.get("author_id", "")),
                     "lang": post.get("lang", ""), "text": post["text"],
                     "hashtags": json.dumps(tags, ensure_ascii=False), "relevant": "", "reason": ""})
    write_csv(DATA / "review.csv", rows,
              ["id", "author_id", "lang", "text", "hashtags", "relevant", "reason"])
    print("Fill relevant=yes/no for every row in data/review.csv; document the reason.")


def clean(text, stop_words):
    """Return one cleaned string using the stopword set supplied for its language.

    Removes URLs, mentions, numbers, punctuation, emojis and single letters.
    Retains hashtag words and German letters. Does not translate or stem words.
    """
    text = unicodedata.normalize("NFKC", html.unescape(text)).casefold()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"(?<!\w)@\w+", " ", text)
    words = re.findall(r"[^\W\d_]+", text)  # Unicode letters, excluding digits/underscore.
    return " ".join(word for word in words if len(word) > 1 and word not in stop_words)


def analyze():
    """Clean reviewed posts, count entities, and generate NMF candidate topics.

    Reads review.csv; writes hashtag/author counts, candidate terms and topic_review.csv.
    German and English are modelled separately; a human must reconcile their labels.
    This stage is offline. Re-running it overwrites analysis outputs, including review labels.
    """
    import numpy as np  # Numerical arrays; used to balance NMF component scales.
    from nltk.corpus import stopwords  # Downloaded German/English stopword lists.
    from sklearn.feature_extraction.text import TfidfVectorizer  # Text-to-number representation.
    from sklearn.decomposition import NMF  # Discover additive groups of weighted terms.

    rows = read_csv(DATA / "review.csv")
    if not rows or any(row["relevant"].strip().lower() not in ("yes", "no") for row in rows):
        raise ValueError("Complete all yes/no relevance decisions first.")
    # Protect any human topic labels already entered into this output.
    if (OUTPUT / "topic_review.csv").exists():
        raise ValueError("Save/move your previous outputs before running analysis again.")
    posts = []
    seen_text = set()
    for row in rows:
        text = " ".join(unicodedata.normalize("NFKC", row["text"]).split())
        if row["relevant"].strip().lower() == "yes" and text not in seen_text:
            posts.append(row)
            seen_text.add(text)  # Exact normalized-text duplicates keep the first author.
    if not posts:
        raise ValueError("No relevant posts remain.")

    # Count entities BEFORE cleaning. A hashtag counts at most once per post.
    authors = Counter(row["author_id"] for row in posts if row["author_id"])
    hashtags = Counter(tag for row in posts for tag in set(json.loads(row["hashtags"])))
    write_csv(OUTPUT / "authors.csv", [{"author_id": key, "posts": n}
              for key, n in authors.most_common()], ["author_id", "posts"])
    write_csv(OUTPUT / "hashtags.csv", [{"hashtag": key, "posts": n}
              for key, n in hashtags.most_common()], ["hashtag", "posts"])

    assignments, topics = [], []
    for code, language in [("de", "german"), ("en", "english")]:
        # Keep selected negations; remove city names that cannot distinguish civic topics.
        stop_words = set(stopwords.words(language)) - {"not", "no", "nor", "nicht", "kein", "keine"}
        stop_words.update({"münchen", "munich", "muenchen"})
        group = [row.copy() for row in posts if row["lang"] == code]
        for row in group:
            row["clean_text"] = clean(row["text"], stop_words)
            row["candidate_topic"] = "unmodelled"
            row["final_topic"] = ""  # Filled by human review, using shared labels across languages.
        usable = [row for row in group if row["clean_text"]]
        if len(usable) >= max(20, TOPICS_PER_LANGUAGE * 2):
            # Keep words/word pairs appearing in >=2 posts and <=95% of this language's posts.
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95)
            matrix = vectorizer.fit_transform([row["clean_text"] for row in usable])
            if min(matrix.shape) < TOPICS_PER_LANGUAGE:
                raise ValueError(f"Not enough vocabulary/posts to model {code} topics.")
            model = NMF(n_components=TOPICS_PER_LANGUAGE, init="nndsvda", random_state=42, max_iter=1000)
            weights = model.fit_transform(matrix)
            # X ≈ W @ H: posts-by-topics weights and topics-by-terms components.
            # Balance factor scales before selecting each post's largest score.
            scores = weights * np.linalg.norm(model.components_, axis=1)
            for index, row in enumerate(usable):
                if scores[index].sum() > 0:
                    row["candidate_topic"] = f"{code}_{scores[index].argmax() + 1}"
            names = vectorizer.get_feature_names_out()
            for topic, component in enumerate(model.components_):
                label = f"{code}_{topic + 1}"
                terms = names[component.argsort()[-10:][::-1]]  # Ten largest term weights.
                topics.append({"candidate_topic": label, "top_terms": ", ".join(terms),
                               "posts": sum(row["candidate_topic"] == label for row in group)})
        else:
            print(f"Too few {code} posts to model. They remain available for manual review.")
        assignments.extend(group)  # Also retain empty/zero-loading posts as unmodelled.

    write_csv(OUTPUT / "candidate_topics.csv", topics, ["candidate_topic", "top_terms", "posts"])
    write_csv(OUTPUT / "topic_review.csv", assignments,
              ["id", "author_id", "lang", "text", "clean_text", "candidate_topic", "final_topic"])
    print(f"Entity counts: {len(posts)} relevant deduplicated posts; topic review: {len(assignments)} de/en posts.")
    print("Inspect texts and terms, then fill final_topic with consistent shared labels.")


def summarize():
    """Count final human-reviewed labels and export the five largest substantive topics.

    Every row needs a label; use other_uncertain for ambiguous posts. Shares use
    ALL topic-review rows, including uncertain ones. Counts describe the sample,
    not Munich residents. Returns nothing; writes top_five.csv locally.
    """
    rows = read_csv(OUTPUT / "topic_review.csv")
    if not rows or any(not row["final_topic"].strip() for row in rows):
        raise ValueError("Fill every final_topic, using other_uncertain where necessary.")
    counts = Counter(row["final_topic"].strip() for row in rows)
    ranked = [(label, count) for label, count in counts.most_common() if label != "other_uncertain"]
    write_csv(OUTPUT / "top_five.csv", [{"topic": label, "posts": count,
              "share": count / len(rows)} for label, count in ranked[:5]], ["topic", "posts", "share"])
    print(f"Denominator: {len(rows)} posts; uncertain: {counts['other_uncertain']}.")
    if len(ranked) < 5:
        print("Fewer than five substantive topics exist; do not invent extra findings.")


# Select just one stage, so re-running analysis cannot trigger collection.
# Examples: python munich_simple.py collect / prepare / analyze / summarize
if __name__ == "__main__":
    stages = {"collect": collect, "prepare": prepare, "analyze": analyze, "summarize": summarize}
    if len(sys.argv) != 2 or sys.argv[1] not in stages:
        raise SystemExit("Usage: python munich_simple.py collect|prepare|analyze|summarize")
    stages[sys.argv[1]]()
