"""Munich civic-discussion research starter. No API call without --execute-api."""
import argparse
import csv
import html
import json
import os
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
QUERY = '(München OR Munich OR Muenchen OR #München OR #Munich) (lang:de OR lang:en) -is:retweet'


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    # IDs must stay strings; spreadsheet software can otherwise round them.
    with Path(path).open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def collect(args):
    start = datetime.fromisoformat(args.start.replace('Z', '+00:00'))
    end = datetime.fromisoformat(args.end.replace('Z', '+00:00'))
    if start.tzinfo is None or end.tzinfo is None or not start < end:
        raise ValueError('Use timezone-aware dates with start earlier than end.')
    if end > datetime.now(timezone.utc):
        raise ValueError('The end date must be in the past.')
    if not 10 <= args.limit <= 10000:
        raise ValueError('Limit must be between 10 and 10000 posts.')
    plan = dict(query=args.query, start=args.start, end=args.end,
                endpoint=args.endpoint, post_limit=args.limit,
                note='Limit bounds returned posts, not a guaranteed monetary cost.')
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if not args.execute_api:
        print('DRY RUN: no API request made. Add --execute-api only when ready to pay.')
        return
    if Path(args.output).exists():
        raise ValueError('Output exists. Choose a new output filename for each collection.')
    import tweepy
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env')
    token = os.environ.get('X_BEARER_TOKEN')
    if not token:
        raise ValueError('Set X_BEARER_TOKEN in .env.')
    client = tweepy.Client(bearer_token=token, wait_on_rate_limit=False)
    method = client.search_recent_tweets if args.endpoint == 'recent' else client.search_all_tweets
    rows, next_token, calls = [], None, 0
    # Requesting no user expansions avoids additional user-resource lookups.
    # No automatic retries: each explicit invocation is a separate collection.
    target = Path(args.output)
    manifest = target.with_suffix('.manifest.json')
    try:
        while len(rows) < args.limit:
            remaining = args.limit - len(rows)
            if remaining < 10:  # API page minimum; never exceed the chosen post cap.
                break
            response = method(
                query=args.query, start_time=args.start, end_time=args.end,
                max_results=min(100, remaining), next_token=next_token,
                tweet_fields=['id', 'text', 'author_id', 'created_at', 'lang',
                              'entities', 'referenced_tweets', 'geo'],
            )
            calls += 1
            for post in response.data or []:
                record = dict(post.data)
                if hasattr(record.get('created_at'), 'isoformat'):
                    record['created_at'] = record['created_at'].isoformat()
                rows.append(record)
            if response.errors:
                raise RuntimeError('API returned partial errors; inspect saved manifest.')
            next_token = (response.meta or {}).get('next_token')
            if not next_token:
                break
    finally:
        save_json(target, rows)
        save_json(manifest, {**plan, 'received': len(rows), 'completed_requests': calls,
                            'next_token': next_token,
                            'retrieved_at': datetime.now(timezone.utc).isoformat()})
    print(f'Saved {len(rows)} posts locally to {target}.')


def prepare(args):
    posts = json.loads(Path(args.input).read_text(encoding='utf-8'))
    rows, seen = [], set()
    for post in posts:
        post_id = str(post['id'])
        if post_id in seen:
            continue
        seen.add(post_id)
        tags = [item['tag'].casefold() for item in
                post.get('entities', {}).get('hashtags', [])]
        rows.append(dict(id=post_id, author_id=str(post.get('author_id', '')),
                         created_at=post.get('created_at', ''), lang=post.get('lang', ''),
                         text=post['text'], hashtags=json.dumps(tags, ensure_ascii=False),
                         relevant='', reason=''))
    target = Path(args.output)
    if target.exists():
        raise ValueError('Review file already exists; choose another name to preserve annotations.')
    write_csv(target, rows, ['id', 'author_id', 'created_at', 'lang', 'text',
                             'hashtags', 'relevant', 'reason'])
    print('Review each row: relevant=yes/no and a reason. Resolve blanks before analysis.')


def clean(text, language, stopwords):
    text = unicodedata.normalize('NFKC', html.unescape(text)).casefold()
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'(?<!\w)@\w+', ' ', text)
    # Keep hashtag words, German letters, and negations; discard the # symbol.
    words = re.findall(r'[^\W\d_]+', text, flags=re.UNICODE)
    return ' '.join(word for word in words if len(word) > 1 and word not in stopwords[language])


def analyze(args):
    import numpy as np
    from nltk.corpus import stopwords
    from sklearn.decomposition import NMF
    from sklearn.feature_extraction.text import TfidfVectorizer
    output = Path(args.output)
    if output.exists():
        raise ValueError('Output directory exists. Choose a new directory for each model run.')
    source = read_csv(args.input)
    if not source or any(row['relevant'].strip().lower() not in ('yes', 'no') for row in source):
        raise ValueError('Complete the relevance review with yes/no before analysis.')
    negations = {'not', 'no', 'nor', 'nicht', 'kein', 'keine', 'keinen', 'keinem', 'keiner'}
    try:
        stops = {lang: set(stopwords.words(name)) - negations
                 for lang, name in [('de', 'german'), ('en', 'english')]}
    except LookupError as error:
        raise ValueError('Run: python -m nltk.downloader stopwords') from error
    # City names are query terms with little power to distinguish civic themes.
    for words in stops.values():
        words.update({'münchen', 'munich', 'muenchen'})
    relevant = [row for row in source if row['relevant'].strip().lower() == 'yes']
    unique, seen_ids, seen_text = [], set(), set()
    duplicates = 0
    for row in relevant:
        key = re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', row['text'])).strip()
        if row['id'] in seen_ids or key in seen_text:
            duplicates += 1
            continue
        seen_ids.add(row['id'])
        seen_text.add(key)
        unique.append(row)
    # Entity counts use retained ORIGINAL text/metadata, before text cleaning.
    authors = Counter(row['author_id'] for row in unique if row['author_id'])
    hashtags = Counter(tag for row in unique for tag in set(json.loads(row['hashtags'])))
    retained = []
    for row in unique:
        row['clean_text'] = clean(row['text'], row['lang'], stops) if row['lang'] in stops else ''
        if row['clean_text']:
            retained.append(row)
    write_csv(output / 'authors.csv',
              [dict(author_id=key, posts=n, share=n / len(unique)) for key, n in authors.most_common()],
              ['author_id', 'posts', 'share'])
    write_csv(output / 'hashtags.csv',
              [dict(hashtag=key, posts=n) for key, n in hashtags.most_common()], ['hashtag', 'posts'])
    topics, assignments, model_info = [], [], {}
    for language in ('de', 'en'):
        group = [row for row in retained if row['lang'] == language]
        if len(group) < max(20, args.topics * 2):
            model_info[language] = {'posts': len(group), 'status': 'Too few posts for this starter model.'}
            for row in group:
                assignments.append({**row, 'candidate_topic': 'unmodelled', 'final_topic': ''})
            continue
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95,
                                     max_features=10000, sublinear_tf=True)
        try:
            matrix = vectorizer.fit_transform([row['clean_text'] for row in group])
        except ValueError:
            raise ValueError(f'No usable {language} vocabulary. Inspect cleaning and min_df.')
        if min(matrix.shape) < args.topics:
            raise ValueError(f'Insufficient {language} vocabulary/posts for requested topic count.')
        model = NMF(n_components=args.topics, init='nndsvda', random_state=args.seed, max_iter=1000)
        weights = model.fit_transform(matrix)
        # Normalize component scale before dominant-topic assignment. These are NOT probabilities.
        scores = weights * np.linalg.norm(model.components_, axis=1)[None, :]
        labels = scores.argmax(axis=1)
        covered = scores.sum(axis=1) > 0
        names = vectorizer.get_feature_names_out()
        for index, row in enumerate(group):
            assignments.append({**row, 'candidate_topic': f'{language}_{labels[index] + 1}'
                                if covered[index] else 'unmodelled', 'final_topic': ''})
        for topic in range(args.topics):
            indices = np.flatnonzero((labels == topic) & covered)
            example_indices = sorted(indices, key=lambda i: scores[i, topic], reverse=True)[:5]
            topics.append(dict(candidate_topic=f'{language}_{topic + 1}',
                               top_terms=', '.join(names[model.components_[topic].argsort()[-12:][::-1]]),
                               assigned_posts=len(indices),
                               representative_ids=';'.join(group[i]['id'] for i in example_indices)))
        model_info[language] = dict(posts=len(group), features=matrix.shape[1],
                                    unmodelled=int((~covered).sum()), iterations=int(model.n_iter_),
                                    iteration_limit_reached=bool(model.n_iter_ >= model.max_iter))
    write_csv(output / 'candidate_topics.csv', topics,
              ['candidate_topic', 'top_terms', 'assigned_posts', 'representative_ids'])
    write_csv(output / 'topic_review.csv', assignments,
              ['id', 'author_id', 'lang', 'text', 'clean_text', 'candidate_topic', 'final_topic'])
    save_json(output / 'run.json', dict(reviewed=len(source), relevant=len(relevant),
              duplicate_texts_or_ids_removed=duplicates, entity_denominator=len(unique),
              model_input_posts=len(retained), excluded_after_cleaning=len(unique)-len(retained),
              topics_per_language=args.topics, seed=args.seed, model_info=model_info,
              input=str(Path(args.input).resolve())))
    print(f'Candidate topics and local review files saved to {output}. Review before interpreting.')


def summarize(args):
    rows = read_csv(args.input)
    if not rows or any(not row.get('final_topic', '').strip() for row in rows):
        raise ValueError('Fill final_topic for EVERY row; use other_uncertain for ambiguous posts.')
    labels = Counter(row['final_topic'].strip() for row in rows)
    ranked = [(label, count) for label, count in labels.most_common() if label != 'other_uncertain']
    write_csv(args.output, [dict(topic=label, posts=count, share_of_reviewed_posts=count/len(rows))
                            for label, count in ranked], ['topic', 'posts', 'share_of_reviewed_posts'])
    print('All reviewed labels exported. Report the top five if at least five defensible topics exist.')
    print(f'Denominator: {len(rows)} posts in topic review; uncertain: {labels["other_uncertain"]}.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('collect')
    p.add_argument('--start', required=True, help='UTC timestamp, e.g. YYYY-MM-DDTHH:MM:SSZ')
    p.add_argument('--end', required=True)
    p.add_argument('--query', default=QUERY)
    p.add_argument('--endpoint', choices=['recent', 'all'], default='recent')
    p.add_argument('--limit', type=int, default=10)
    p.add_argument('--output', default='data/raw.json')
    p.add_argument('--execute-api', action='store_true')
    p.set_defaults(action=collect)
    p = sub.add_parser('prepare')
    p.add_argument('--input', default='data/raw.json')
    p.add_argument('--output', default='data/relevance_review.csv')
    p.set_defaults(action=prepare)
    p = sub.add_parser('analyze')
    p.add_argument('--input', default='data/relevance_review.csv')
    p.add_argument('--output', default='outputs/baseline')
    p.add_argument('--topics', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    p.set_defaults(action=analyze)
    p = sub.add_parser('summarize')
    p.add_argument('--input', default='outputs/baseline/topic_review.csv')
    p.add_argument('--output', default='outputs/final_topic_counts.csv')
    p.set_defaults(action=summarize)
    args = parser.parse_args()
    if getattr(args, 'topics', 1) < 2:
        parser.error('--topics must be at least 2')
    args.action(args)


if __name__ == '__main__':
    main()
