"""Stubs for external data collection respecting ToS.
Fill in credentials and enable only if compliant for your account and region.
"""
from pathlib import Path
import json
import tweepy

ROOT = Path(__file__).resolve().parents[1]
RAW_EXT = ROOT / "data" / "raw" / "external"
RAW_EXT.mkdir(parents=True, exist_ok=True)


def save_jsonlines(path: Path, rows: list[dict]):
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def collect_x_hashtags(hashtag: str, max_items: int = 200):
    """Collect recent tweets for a hashtag using the X API v2.
    Requires a valid Bearer Token in your environment or config.
    """
    bearer_token = "AAAAAAAAAAAAAAAAAAAAAL4I4AEAAAAALhf7EhCH5TFUcD9a9mle0YIQMbE%3DDfgdcJuHjfZ261K1BjXWvJpTzEXgh7njsZlVIc6YBQf3qmI5FB"

    client = tweepy.Client(
        bearer_token=bearer_token,
        wait_on_rate_limit=True
    )

    query = f"#{hashtag} -is:retweet"   
    tweets = client.search_recent_tweets(
        query=query,
        tweet_fields=["id", "text", "created_at", "public_metrics", "lang"],
        max_results=100  
    )

    rows = []
    if tweets.data:
        for t in tweets.data[:max_items]:
            rows.append({
                "id": t.id,
                "created_at": str(t.created_at),
                "text": t.text,
                "lang": t.lang,
                "metrics": t.public_metrics
            })

    out_path = RAW_EXT / f"{hashtag}.jsonl"
    save_jsonlines(out_path, rows)
    print(f"Saved {len(rows)} tweets → {out_path}")


if __name__ == "__main__":
    # Example dry run
    collect_x_hashtags("BeautyTok", max_items=10)
