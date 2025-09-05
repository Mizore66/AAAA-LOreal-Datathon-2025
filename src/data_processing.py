from pathlib import Path
from typing import List, Optional, Dict, Tuple
import re
import pandas as pd
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
INTERIM_DIR = ROOT / "data" / "interim"

INTERIM_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Text cleaning utilities
# -----------------------------

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"@[\w_]+", re.IGNORECASE)
# Keep hashtags, remove most punctuation except '#'
NON_ALNUM_RE = re.compile(r"[^\w\s#]", re.UNICODE)
MULTISPACE_RE = re.compile(r"\s+")
# Basic emoji ranges removal (not exhaustive but effective)
EMOJI_RE = re.compile(
    "[\U0001F600-\U0001F64F]"  # emoticons
    "|[\U0001F300-\U0001F5FF]"  # symbols & pictographs
    "|[\U0001F680-\U0001F6FF]"  # transport & map symbols
    "|[\U0001F1E0-\U0001F1FF]"  # flags (iOS)
    , flags=re.UNICODE
)

HASHTAG_RE = re.compile(r"#(\w+)")

STOPWORDS = set(
    "the a an and or of in to for with on at from by is are was were be been being this that these those it its as if then than also not no but so very can will just into over under above below you your we our their they i me my mine out up down across about after before during between within without more less".split()
)

KEYWORDS = [
    # Skincare ingredients & actives
    "hyaluronic acid",
    "niacinamide",
    "salicylic acid",
    "glycolic acid",
    "lactic acid",
    "azelaic acid",
    "tranexamic acid",
    "benzoyl peroxide",
    "retinol",
    "retinoid",
    "bakuchiol",
    "vitamin c",
    "vitamin e",
    "panthenol",
    "ceramide",
    "ceramides",
    "peptides",
    "copper peptides",
    "squalane",
    "allantoin",
    "urea",
    "snail mucin",
    "centella",
    "cica",
    "tea tree",
    # Skincare concepts & routines
    "skin barrier",
    "barrier repair",
    "double cleanse",
    "oil cleanser",
    "exfoliation",
    "chemical exfoliant",
    "physical exfoliant",
    "moisturizer",
    "sunscreen",
    "spf",
    "spf 50",
    "reapply spf",
    "skin cycling",
    "skin flooding",
    "slugging",
    "glass skin",
    "dewy skin",
    "matte skin",
    "non comedogenic",
    "fragrance free",
    "cruelty free",
    "vegan",
    "clean beauty",
    "k beauty",
    "j beauty",
    # Makeup
    "foundation",
    "concealer",
    "blush",
    "cream blush",
    "mascara",
    "eyeliner",
    "lipstick",
    "lip oil",
    "lip gloss",
    "lip tint",
    "setting spray",
    "contour",
    "highlighter",
    "bronzer",
    "eyeshadow",
    "brow gel",
    "no makeup makeup",
    "soft glam",
    "latte makeup",
    "underpainting",
    # Hair
    "hair mask",
    "heat protectant",
    "leave in",
    "bond builder",
    "scalp serum",
    "shampoo",
    "conditioner",
    "sulfate free",
    "keratin",
    "argan oil",
    "rosemary oil",
    "hair growth",
    # General
    "glow",
    "makeup",
    "skincare",
]

# Map select keywords to high-level categories for downstream filtering and visuals
KEYWORD_CATEGORY: Dict[str, str] = {}
for kw in [
    "hyaluronic acid", "niacinamide", "salicylic acid", "glycolic acid", "lactic acid", "azelaic acid",
    "tranexamic acid", "benzoyl peroxide", "retinol", "retinoid", "bakuchiol", "vitamin c", "vitamin e",
    "panthenol", "ceramide", "ceramides", "peptides", "copper peptides", "squalane", "allantoin", "urea",
    "snail mucin", "centella", "cica", "tea tree", "skin barrier", "barrier repair", "double cleanse",
    "oil cleanser", "exfoliation", "chemical exfoliant", "physical exfoliant", "moisturizer", "sunscreen",
    "spf", "spf 50", "reapply spf", "skin cycling", "skin flooding", "slugging", "glass skin", "dewy skin",
    "matte skin", "non comedogenic", "fragrance free", "cruelty free", "vegan", "clean beauty", "k beauty",
    "j beauty", "skincare"
]:
    KEYWORD_CATEGORY[kw] = "Skincare"

for kw in [
    "foundation", "concealer", "blush", "cream blush", "mascara", "eyeliner", "lipstick", "lip oil",
    "lip gloss", "lip tint", "setting spray", "contour", "highlighter", "bronzer", "eyeshadow", "brow gel",
    "no makeup makeup", "soft glam", "latte makeup", "underpainting", "makeup"
]:
    KEYWORD_CATEGORY[kw] = "Makeup"

for kw in [
    "hair mask", "heat protectant", "leave in", "bond builder", "scalp serum", "shampoo", "conditioner",
    "sulfate free", "keratin", "argan oil", "rosemary oil", "hair growth"
]:
    KEYWORD_CATEGORY[kw] = "Hair"


def categorize_feature(feature: str) -> str:
    """Return a high-level category for a feature token (hashtag, keyword, audio ID/title).
    For hashtags, we strip the leading '#'. Default is 'Other'.
    """
    if not isinstance(feature, str):
        return "Other"
    f = feature.lstrip('#').lower()
    # Direct keyword match
    if f in KEYWORD_CATEGORY:
        return KEYWORD_CATEGORY[f]
    # Heuristic contains-based match
    for k, cat in KEYWORD_CATEGORY.items():
        if k in f:
            return cat
    # Hair/Makeup/Skincare hint tokens
    if any(tok in f for tok in ["skin", "spf", "sunscreen", "barrier", "niacinamide", "retinol", "serum"]):
        return "Skincare"
    if any(tok in f for tok in ["lip", "lash", "brow", "blush", "contour", "eyeshadow", "mascara", "liner"]):
        return "Makeup"
    if any(tok in f for tok in ["hair", "scalp", "shampoo", "conditioner", "keratin", "oil"]):
        return "Hair"
    return "Other"


def clean_text(text: Optional[str]) -> str:
    if not isinstance(text, str):
        return ""
    s = text.lower()
    s = URL_RE.sub(" ", s)
    s = MENTION_RE.sub(" ", s)
    s = EMOJI_RE.sub(" ", s)
    s = NON_ALNUM_RE.sub(" ", s)
    s = MULTISPACE_RE.sub(" ", s).strip()
    return s


def extract_hashtags(text: str) -> List[str]:
    return [m.group(1).lower() for m in HASHTAG_RE.finditer(text or "") if m.group(1)]


def find_timestamp_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [c for c in df.columns if any(k in c.lower() for k in ["time", "date", "created", "upload", "posted"])]
    for c in candidates:
        try:
            pd.to_datetime(df[c])
            return c
        except Exception:
            continue
    return None


def find_text_columns(df: pd.DataFrame) -> List[str]:
    # Common text fields in social/video/comment datasets
    preferred = ["text", "comment", "caption", "title", "description", "body", "content"]
    cols = [c for c in preferred if c in df.columns]
    # Fallback: any object dtype columns
    if not cols:
        cols = [c for c in df.columns if df[c].dtype == 'object'][:2]
    return cols


def find_audio_column(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        cl = c.lower()
        if any(k in cl for k in ["audio", "sound", "music", "track_id", "song_id"]):
            return c
    return None


def load_samples() -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    # Prefer full files over samples
    files = [p for p in PROC_DIR.glob('*.parquet') if not p.name.endswith('.sample.parquet')]
    if not files:
        files = list(PROC_DIR.glob('*.parquet'))
    print(f"[Phase2] Loading processed parquet files (excluding samples where possible): {len(files)} files")
    for p in files:
        name = p.stem
        try:
            out[name] = pd.read_parquet(p)
            print(f"  - loaded {p.name} (rows={len(out[name])}, cols={len(out[name].columns)})")
        except Exception as e:
            print(f"[WARN] Failed to load {p}: {e}")
    return out


def _prepare_text_df(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    ts_col = find_timestamp_column(df)
    if not ts_col:
        return None
    txt_cols = find_text_columns(df)
    if not txt_cols:
        return None
    use = df[[ts_col] + txt_cols].copy()
    use["ts"] = pd.to_datetime(use[ts_col], errors='coerce')
    use = use.dropna(subset=["ts"]).copy()
    # merge text columns
    use["text_raw"] = use[txt_cols].astype(str).agg(" ".join, axis=1)
    use["text_clean"] = use["text_raw"].map(clean_text)
    use["hashtags"] = use["text_raw"].map(lambda s: extract_hashtags(s))
    return use[["ts", "text_clean", "hashtags"]]


def aggregate_hashtags_6h(dfs: List[Tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    frames = []
    for name, df in tqdm(dfs, desc="Aggregating hashtags", unit="src"):
        p = _prepare_text_df(df)
        if p is None or p.empty:
            continue
        # explode hashtags
        pe = p[["ts", "hashtags"]].explode("hashtags")
        pe = pe.dropna(subset=["hashtags"])  # remove rows without hashtags
    pe["bin"] = pe["ts"].dt.floor('6h')
    g = pe.groupby(["bin", "hashtags"], as_index=False).size().rename(columns={"size": "count"})
    g = g.rename(columns={"hashtags": "feature"})
    g["category"] = g["feature"].map(categorize_feature)
    frames.append(g)
    if not frames:
        return pd.DataFrame(columns=["bin", "feature", "count", "rolling_mean_24h", "delta_vs_mean"])
    allg = pd.concat(frames, ignore_index=True)
    allg = allg.groupby(["bin", "feature", "category"], as_index=False)["count"].sum()
    # rolling per feature across time
    allg = allg.sort_values(["feature", "bin"]).reset_index(drop=True)
    allg["rolling_mean_24h"] = (
        allg.groupby("feature")["count"].transform(lambda s: s.rolling(window=4, min_periods=1).mean())
    )
    allg["delta_vs_mean"] = allg["count"] - allg["rolling_mean_24h"]
    return allg


def aggregate_keywords_6h(dfs: List[Tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    frames = []
    kw_patterns = [(kw, re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE)) for kw in KEYWORDS]
    for name, df in tqdm(dfs, desc="Aggregating keywords", unit="src"):
        p = _prepare_text_df(df)
        if p is None or p.empty:
            continue
        p["bin"] = p["ts"].dt.floor('6h')
        for kw, pat in kw_patterns:
            mask = p["text_clean"].str.contains(pat)
            g = p.loc[mask].groupby("bin", as_index=False).size().rename(columns={"size": "count"})
            g["feature"] = kw
            g["category"] = categorize_feature(kw)
            frames.append(g)
    if not frames:
        return pd.DataFrame(columns=["bin", "feature", "count", "rolling_mean_24h", "delta_vs_mean"])
    allg = pd.concat(frames, ignore_index=True)
    allg = allg.groupby(["bin", "feature", "category"], as_index=False)["count"].sum()
    allg = allg.sort_values(["feature", "bin"]).reset_index(drop=True)
    allg["rolling_mean_24h"] = (
        allg.groupby("feature")["count"].transform(lambda s: s.rolling(window=4, min_periods=1).mean())
    )
    allg["delta_vs_mean"] = allg["count"] - allg["rolling_mean_24h"]
    return allg


def aggregate_audio_6h(dfs: List[Tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for name, df in tqdm(dfs, desc="Aggregating audio", unit="src"):
        ts_col = find_timestamp_column(df)
        audio_col = find_audio_column(df)
        if not ts_col or not audio_col:
            continue
        use = df[[ts_col, audio_col]].copy()
        use["ts"] = pd.to_datetime(use[ts_col], errors='coerce')
        use = use.dropna(subset=["ts", audio_col])
        if use.empty:
            continue
        use["bin"] = use["ts"].dt.floor('6h')
        g = use.groupby(["bin", audio_col], as_index=False).size().rename(columns={"size": "count"})
        g = g.rename(columns={audio_col: "feature"})
        g["category"] = "Other"
        frames.append(g)
    if not frames:
        return pd.DataFrame(columns=["bin", "feature", "count", "rolling_mean_24h", "delta_vs_mean"])
    allg = pd.concat(frames, ignore_index=True)
    allg = allg.groupby(["bin", "feature", "category"], as_index=False)["count"].sum()
    allg = allg.sort_values(["feature", "bin"]).reset_index(drop=True)
    allg["rolling_mean_24h"] = (
        allg.groupby("feature")["count"].transform(lambda s: s.rolling(window=4, min_periods=1).mean())
    )
    allg["delta_vs_mean"] = allg["count"] - allg["rolling_mean_24h"]
    return allg


def write_parquet(df: pd.DataFrame, name: str) -> Path:
    out = PROC_DIR / name
    df.to_parquet(out, index=False)
    print(f"Wrote {out}")
    return out


def write_phase2_report(h_df: pd.DataFrame, k_df: pd.DataFrame, a_df: pd.DataFrame):
    path = INTERIM_DIR / 'phase2_features_report.md'
    lines = ["# Phase 2 Feature Engineering Report", ""]
    # Overall counts
    lines.append("## Summary")
    lines.append(f"- Hashtag rows: {len(h_df)}")
    lines.append(f"- Keyword rows: {len(k_df)}")
    lines.append(f"- Audio rows: {len(a_df)}")
    lines.append("")
    for label, df in [("Hashtags", h_df), ("Keywords", k_df), ("Audio", a_df)]:
        lines.append(f"## {label}")
        if df.empty:
            lines.append("No data available.")
            lines.append("")
            continue
        # Latest window top anomalies by delta_vs_mean
        latest_bin = df['bin'].max()
        latest = df[df['bin'] == latest_bin].sort_values('delta_vs_mean', ascending=False).head(15)
        try:
            lines.append(f"Latest bin: {latest_bin}")
            cols = [c for c in ["category", "feature", "count", "rolling_mean_24h", "delta_vs_mean"] if c in latest.columns]
            lines.append(latest[cols].to_markdown(index=False))
        except Exception as e:
            lines.append(f"(Could not render table: {e})")
        lines.append("")
        # Last 7 days (if available)
        try:
            start_7d = latest_bin - pd.Timedelta(days=7)
            last7 = df[(df['bin'] >= start_7d) & (df['bin'] <= latest_bin)]
            if not last7.empty:
                top7 = last7.groupby(['feature', 'category'], as_index=False)['count'].sum().sort_values('count', ascending=False).head(15)
                lines.append("Top features in last 7 days:")
                lines.append(top7.to_markdown(index=False))
                lines.append("")
        except Exception as e:
            lines.append(f"(Could not render last-7-days table: {e})")
    # Category-level summary for latest bin across all sources
    lines.append("## Category summary (latest bin)")
    latest_rows = []
    for df in [h_df, k_df, a_df]:
        if df.empty:
            continue
        lb = df['bin'].max()
        latest_rows.append(df[df['bin'] == lb])
    if latest_rows:
        latest_all = pd.concat(latest_rows, ignore_index=True)
        try:
            cat_summary = latest_all.groupby('category', as_index=False)['count'].sum().sort_values('count', ascending=False)
            lines.append(cat_summary.to_markdown(index=False))
        except Exception as e:
            lines.append(f"(Could not render category summary: {e})")
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"Report written to {path}")


def main():
    samples = load_samples()
    # choose likely comment/video tables by name heuristic
    text_sources = []
    for name, df in samples.items():
        if any(tok in name for tok in ["comment", "video"]):
            text_sources.append((name, df))
    if not text_sources:
        text_sources = list(samples.items())

    print("[Phase2] Aggregations starting...")
    ts_hashtags = aggregate_hashtags_6h(text_sources)
    ts_keywords = aggregate_keywords_6h(text_sources)
    ts_audio = aggregate_audio_6h(text_sources)
    print("[Phase2] Aggregations completed.")

    write_parquet(ts_hashtags, 'features_hashtags_6h.parquet')
    write_parquet(ts_keywords, 'features_keywords_6h.parquet')
    write_parquet(ts_audio, 'features_audio_6h.parquet')

    write_phase2_report(ts_hashtags, ts_keywords, ts_audio)


if __name__ == '__main__':
    main()
