# External Data Guidelines

- Respect Terms of Service. Prefer official APIs; if scraping, use public metadata only.
- Rate limit responsibly and avoid personal identifiers.
- Store any collected raw files in `data/raw/external/` with source, date, and query in filename.
- Keep a `data_dictionary.md` documenting fields for each source.

## Suggested sources
- X (Twitter) API v2: tweets with beauty hashtags.
- Instagram Graph API (if access): hashtag insights.
- Public datasets (e.g., Kaggle TikTok trends).

## Lightweight collectors (optional)
- `snscrape` for X as a fallback (metadata only). Example script scaffolding is in `src/external_sources.py`.
