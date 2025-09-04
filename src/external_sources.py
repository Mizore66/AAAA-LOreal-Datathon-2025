"""Stubs for external data collection respecting ToS.
Fill in credentials and enable only if compliant for your account and region.
"""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
RAW_EXT = ROOT / "data" / "raw" / "external"
RAW_EXT.mkdir(parents=True, exist_ok=True)


def save_jsonlines(path: Path, rows: list[dict]):
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def collect_x_hashtags(hashtag: str, max_items: int = 200):
    """Placeholder for X (Twitter) API v2 search.
    Implement with official API. Do not scrape personal data.
    """
    # TODO: Integrate tweepy / requests with bearer token


if __name__ == "__main__":
    # Example dry run
    collect_x_hashtags("BeautyTok", max_items=10)
