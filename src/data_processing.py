from pathlib import Path
from typing import List, Optional, Dict, Tuple, Set
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
from collections import Counter, defaultdict
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# -----------------------------
# Performance / Fast Mode Controls
# -----------------------------
FAST_MODE = False                # Global switch to reduce computation cost
MAX_FEATURES_EMERGING = 400      # Cap emerging term features per timeframe
MAX_FEATURES_ANOMALY = 400       # Cap features for statistical anomaly scan
MAX_CLUSTER_FEATURES = 250       # Cap features for clustering
COMPUTE_VELOCITY = False if FAST_MODE else True  # Skip velocity in fast mode (except 6h primary)
SAMPLE_ROWS_PER_SOURCE = 150_000 # If a raw text source exceeds this, sample rows

from itertools import combinations

# Optional heavy audio feature extraction (graceful fallback)
try:
    import librosa  # type: ignore
    HAS_LIBROSA = True
except Exception:
    HAS_LIBROSA = False

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed" / "dataset"
INTERIM_DIR = ROOT / "data" / "interim"
CACHE_DIR = ROOT / "data" / "cache"

# Create directories
for dir_path in [INTERIM_DIR, CACHE_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

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


# -----------------------------
# Enhanced Trend Detection (Phase 3)
# -----------------------------

def get_cache_path(name: str) -> Path:
    """Get cache file path for a given cache name."""
    return CACHE_DIR / f"{name}.pkl"


def load_cache(name: str) -> Optional[dict]:
    """Load cached data if it exists and is recent."""
    cache_path = get_cache_path(name)
    if cache_path.exists():
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"[WARN] Failed to load cache {name}: {e}")
    return None


def save_cache(name: str, data: dict) -> None:
    """Save data to cache."""
    cache_path = get_cache_path(name)
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        print(f"[WARN] Failed to save cache {name}: {e}")


def extract_ngrams(text: str, n: int = 2) -> List[str]:
    """Extract n-grams from cleaned text."""
    if not isinstance(text, str) or not text.strip():
        return []
    
    words = text.split()
    # Filter out very short words and stopwords
    words = [w for w in words if len(w) >= 3 and w.lower() not in STOPWORDS]
    
    if len(words) < n:
        return []
    
    ngrams = []
    for i in range(len(words) - n + 1):
        ngram = " ".join(words[i:i+n])
        ngrams.append(ngram)
    
    return ngrams


def extract_all_terms(text: str, min_length: int = 3) -> List[str]:
    """Extract all meaningful terms from text (words + phrases)."""
    if not isinstance(text, str):
        return []
    
    terms = []
    words = text.split()
    
    # Add individual words (filtered)
    for word in words:
        if len(word) >= min_length and word.lower() not in STOPWORDS:
            terms.append(word)
    
    # Add bigrams and trigrams
    terms.extend(extract_ngrams(text, 2))
    terms.extend(extract_ngrams(text, 3))
    
    return terms


def calculate_term_velocity(term_counts: pd.DataFrame, term: str, window_hours: int = 24) -> float:
    """Calculate how quickly a term is gaining traction (velocity)."""
    term_data = term_counts[term_counts['feature'] == term].sort_values('bin')
    
    if len(term_data) < 2:
        return 0.0
    
    # Get recent data within the window
    latest_time = term_data['bin'].max()
    start_time = latest_time - pd.Timedelta(hours=window_hours)
    recent_data = term_data[term_data['bin'] >= start_time]
    
    if len(recent_data) < 2:
        return 0.0
    
    # Calculate velocity as rate of change
    x = np.arange(len(recent_data))
    y = recent_data['count'].values
    
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    
    # Simple linear regression slope
    slope, _, _, _, _ = stats.linregress(x, y)
    return slope


def detect_statistical_anomalies(term_counts: pd.DataFrame, z_threshold: float = 2.5) -> pd.DataFrame:
    """Detect statistically anomalous spikes in term frequency."""
    anomalies = []
    features = term_counts['feature'].unique()
    # Fast mode cap
    if FAST_MODE and len(features) > MAX_FEATURES_ANOMALY:
        features = features[:MAX_FEATURES_ANOMALY]
    for term in features:
        term_data = term_counts[term_counts['feature'] == term].sort_values('bin')
        
        if len(term_data) < 5:  # Need enough data points
            continue
        
        counts = term_data['count'].values
        
        # Calculate z-scores for recent points
        mean_count = np.mean(counts[:-2]) if len(counts) > 2 else np.mean(counts)
        std_count = np.std(counts[:-2]) if len(counts) > 2 and np.std(counts[:-2]) > 0 else np.std(counts)
        
        if std_count == 0:
            continue
        
        # Check the most recent points
        for i in range(max(1, len(counts) - 3), len(counts)):
            z_score = (counts[i] - mean_count) / std_count
            
            if z_score > z_threshold:
                anomalies.append({
                    'feature': term,
                    'bin': term_data.iloc[i]['bin'],
                    'count': counts[i],
                    'z_score': z_score,
                    'baseline_mean': mean_count,
                    'velocity': calculate_term_velocity(term_counts, term),
                    'category': categorize_feature(term),
                    'type': 'statistical_anomaly'
                })
    
    return pd.DataFrame(anomalies)


def aggregate_emerging_terms_6h(dfs: List[Tuple[str, pd.DataFrame]], 
                               min_frequency: int = 5,
                               min_growth_rate: float = 2.0) -> pd.DataFrame:
    """Aggregate all meaningful terms and identify emerging ones without keyword filtering."""
    
    # Check cache first
    cache_key = f"emerging_terms_{len(dfs)}_{min_frequency}_{min_growth_rate}"
    cached = load_cache(cache_key)
    
    frames = []
    all_terms_by_time = defaultdict(Counter)
    
    print("[Phase3] Extracting all terms from text data...")
    
    for name, df in tqdm(dfs, desc="Processing emerging terms", unit="src"):
        if FAST_MODE and len(df) > SAMPLE_ROWS_PER_SOURCE:
            df = df.sample(SAMPLE_ROWS_PER_SOURCE, random_state=42)
        p = _prepare_text_df(df)
        if p is None or p.empty:
            continue
        
        # Extract all terms (not just keywords)
        p["all_terms"] = p["text_clean"].map(extract_all_terms)
        p["bin"] = p["ts"].dt.floor('6h')
        
        # Explode terms and count by time bin
        pe = p[["bin", "all_terms"]].explode("all_terms")
        pe = pe.dropna(subset=["all_terms"])
        pe = pe.rename(columns={"all_terms": "feature"})
        
        # Group by time bin and term
        g = pe.groupby(["bin", "feature"], as_index=False).size().rename(columns={"size": "count"})
        frames.append(g)
    
    if not frames:
        return pd.DataFrame(columns=["bin", "feature", "count", "growth_rate", "is_emerging"])
    
    # Combine all data
    allg = pd.concat(frames, ignore_index=True)
    allg = allg.groupby(["bin", "feature"], as_index=False)["count"].sum()
    
    # Filter out very rare terms
    total_counts = allg.groupby("feature")["count"].sum()
    frequent_terms = total_counts[total_counts >= min_frequency].index
    allg = allg[allg["feature"].isin(frequent_terms)]
    
    if allg.empty:
        return pd.DataFrame(columns=["bin", "feature", "count", "growth_rate", "is_emerging"])
    
    # Calculate growth rates (vectorized)
    allg = allg.sort_values(["feature", "bin"]).reset_index(drop=True)
    allg['prev_count'] = allg.groupby('feature')['count'].shift(1)
    allg['growth_rate'] = allg['count'] / (allg['prev_count'] + 1e-6)
    allg.loc[allg['prev_count'].isna(), 'growth_rate'] = 1.0
    
    # Identify emerging terms (high recent growth)
    allg["is_emerging"] = allg["growth_rate"] >= min_growth_rate
    allg["category"] = allg["feature"].map(categorize_feature)
    
    # Add velocity calculations
    if COMPUTE_VELOCITY:
        print("[Phase3] Calculating term velocities...")
        velocities = {}
        for term in tqdm(allg["feature"].unique(), desc="Computing velocities"):
            velocities[term] = calculate_term_velocity(allg, term)
        allg["velocity"] = allg["feature"].map(velocities)
    else:
        allg["velocity"] = 0.0
    
    # Cache results
    save_cache(cache_key, {"data": allg.to_dict(), "timestamp": pd.Timestamp.now()})
    
    return allg

# -- Inserted: generic emerging terms timeframe function --

def aggregate_emerging_terms_timeframe(dfs: List[Tuple[str, pd.DataFrame]], label: str,
                                       min_growth_rate: float = 2.0) -> pd.DataFrame:
    if label == '6h':
        return aggregate_emerging_terms_6h(dfs, min_frequency=_MIN_FREQ_BY_LABEL.get(label, 3), min_growth_rate=min_growth_rate)
    min_frequency = _MIN_FREQ_BY_LABEL.get(label, 5)
    frames = []
    print(f"[Phase3][Emerging:{label}] Extracting all terms...")
    for name, df in tqdm(dfs, desc=f"Emerging terms {label}", unit="src"):
        if FAST_MODE and len(df) > SAMPLE_ROWS_PER_SOURCE:
            df = df.sample(SAMPLE_ROWS_PER_SOURCE, random_state=42)
        p = _prepare_text_df(df)
        if p is None or p.empty:
            continue
        p['all_terms'] = p['text_clean'].map(extract_all_terms)
        p['bin'] = _assign_time_bin(p['ts'], label)
        pe = p[['bin','all_terms']].explode('all_terms').dropna(subset=['all_terms'])
        if pe.empty:
            continue
        pe = pe.rename(columns={'all_terms':'feature'})
        g = pe.groupby(['bin','feature'], as_index=False).size().rename(columns={'size':'count'})
        frames.append(g)
    if not frames:
        return pd.DataFrame(columns=["bin","feature","count","growth_rate","is_emerging","velocity","category"])    
    allg = pd.concat(frames, ignore_index=True)
    allg = allg.groupby(['bin','feature'], as_index=False)['count'].sum()
    total_counts = allg.groupby('feature')['count'].sum()
    frequent = total_counts[total_counts >= min_frequency].index
    allg = allg[allg['feature'].isin(frequent)]
    if allg.empty:
        return pd.DataFrame(columns=["bin","feature","count","growth_rate","is_emerging","velocity","category"])    
    allg = allg.sort_values(['feature','bin']).reset_index(drop=True)
    allg['prev_count'] = allg.groupby('feature')['count'].shift(1)
    allg['growth_rate'] = allg['count'] / (allg['prev_count'] + 1e-6)
    allg.loc[allg['prev_count'].isna(), 'growth_rate'] = 1.0
    allg['is_emerging'] = allg['growth_rate'] >= min_growth_rate
    allg['category'] = allg['feature'].map(categorize_feature)
    if COMPUTE_VELOCITY and label == '6h':
        print(f"[Phase3][Emerging:{label}] Calculating velocities...")
        velocities = {}
        for term in allg['feature'].unique():
            velocities[term] = calculate_term_velocity(allg, term, window_hours=24)
        allg['velocity'] = allg['feature'].map(velocities)
    else:
        allg['velocity'] = 0.0
    return allg


# -----------------------------
# Multi-timeframe aggregation extensions
# -----------------------------

# ---- 6h baseline aggregation helpers (previously assumed but not defined) ----

def _aggregate_baseline(df_pairs: List[Tuple[str, pd.DataFrame]], kind: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    if kind == 'hashtags' or kind == 'keywords':
        kw_patterns = None
        if kind == 'keywords':
            kw_patterns = [(kw, re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE)) for kw in KEYWORDS]
    for name, df in tqdm(df_pairs, desc=f"Aggregating {kind} (6h)", unit="src"):
        if kind in ('hashtags', 'keywords'):
            p = _prepare_text_df(df)
            if p is None or p.empty:
                continue
            p['bin'] = p['ts'].dt.floor('6H')
            if kind == 'hashtags':
                pe = p[['bin','hashtags']].explode('hashtags').dropna(subset=['hashtags'])
                if pe.empty:
                    continue
                g = pe.groupby(['bin','hashtags'], as_index=False).size().rename(columns={'size':'count'})
                g = g.rename(columns={'hashtags':'feature'})
                g['category'] = g['feature'].map(categorize_feature)
                frames.append(g)
            else:  # keywords
                for kw, pat in kw_patterns:  # type: ignore
                    mask = p['text_clean'].str.contains(pat)
                    if not mask.any():
                        continue
                    g = p.loc[mask].groupby('bin', as_index=False).size().rename(columns={'size':'count'})
                    g['feature'] = kw
                    g['category'] = categorize_feature(kw)
                    frames.append(g)
        elif kind == 'audio':
            ts_col = find_timestamp_column(df)
            audio_col = find_audio_column(df)
            if not ts_col or not audio_col:
                continue
            use = df[[ts_col, audio_col]].copy()
            use['ts'] = pd.to_datetime(use[ts_col], errors='coerce')
            use = use.dropna(subset=['ts', audio_col])
            if use.empty:
                continue
            use['bin'] = use['ts'].dt.floor('6H')
            g = use.groupby(['bin', audio_col], as_index=False).size().rename(columns={'size':'count'})
            g = g.rename(columns={audio_col:'feature'})
            # Attempt optional audio feature extraction summarization
            g['category'] = 'Other'
            frames.append(g)
    if not frames:
        return pd.DataFrame(columns=['bin','feature','count','rolling_mean_24h','delta_vs_mean','category'])
    allg = pd.concat(frames, ignore_index=True)
    if kind != 'keywords':  # keywords already have category assignment
        # Ensure category present
        if 'category' not in allg.columns:
            allg['category'] = allg['feature'].map(categorize_feature)
    allg = allg.groupby(['bin','feature','category'], as_index=False)['count'].sum()
    # Rolling stats (24h == 4 * 6h)
    allg = allg.sort_values(['feature','bin']).reset_index(drop=True)
    allg['rolling_mean_24h'] = allg.groupby('feature')['count'].transform(lambda s: s.rolling(window=4, min_periods=1).mean())
    allg['delta_vs_mean'] = allg['count'] - allg['rolling_mean_24h']
    return allg


def aggregate_hashtags_6h(dfs: List[Tuple[str,pd.DataFrame]]) -> pd.DataFrame:
    return _aggregate_baseline(dfs, 'hashtags')


def aggregate_keywords_6h(dfs: List[Tuple[str,pd.DataFrame]]) -> pd.DataFrame:
    return _aggregate_baseline(dfs, 'keywords')


def aggregate_audio_6h(dfs: List[Tuple[str,pd.DataFrame]]) -> pd.DataFrame:
    return _aggregate_baseline(dfs, 'audio')

TIMEFRAME_LABELS = [
    '1h','3h','6h','1d','3d','7d','14d','1m','3m','6m'
]

_FREQ_MAP = {
    '1h': '1H',
    '3h': '3H',
    '6h': '6H',
    '1d': '1D',
    '3d': '3D',
    '7d': '7D',
    '14d': '14D'
    # month-based handled separately
}

_ROLLING_WINDOW = {
    # Window sizes chosen to approximate similar historical coverage across granularities
    '1h': 24,   # past day
    '3h': 8,    # past day
    '6h': 4,    # existing behavior (24h)
    '1d': 7,    # past week
    '3d': 7,    # ~3 weeks
    '7d': 4,    # ~4 weeks
    '14d': 4,   # ~8 weeks
    '1m': 6,    # past 6 months
    '3m': 4,    # past year
    '6m': 4     # past 2 years
}

# Minimum frequency thresholds per timeframe for emerging term filtering
_MIN_FREQ_BY_LABEL = {
    '1h': 3,
    '3h': 3,
    '6h': 3,
    '1d': 5,
    '3d': 5,
    '7d': 8,
    '14d': 10,
    '1m': 15,
    '3m': 20,
    '6m': 30
}

def _assign_time_bin(ts: pd.Series, label: str) -> pd.Series:
    if label in _FREQ_MAP:
        return ts.dt.floor(_FREQ_MAP[label])
    if label == '1m':
        return ts.dt.to_period('M').dt.to_timestamp()
    if label == '3m':  # quarter
        return ts.dt.to_period('Q').dt.to_timestamp()
    if label == '6m':  # half-year custom
        years = ts.dt.year
        months = ts.dt.month
        start_month = np.where(months <= 6, 1, 7)
        return pd.to_datetime({'year': years, 'month': start_month, 'day': 1})
    raise ValueError(f"Unsupported timeframe label: {label}")

def _add_rolling_stats(df: pd.DataFrame, label: str) -> pd.DataFrame:
    if df.empty:
        return df
    window = _ROLLING_WINDOW.get(label, 4)
    df = df.sort_values(["feature","bin"]).reset_index(drop=True)
    df["rolling_mean_24h"] = (
        df.groupby("feature")["count"].transform(lambda s: s.rolling(window=window, min_periods=1).mean())
    )
    df["delta_vs_mean"] = df["count"] - df["rolling_mean_24h"]
    return df

def aggregate_hashtags_timeframe(dfs: List[Tuple[str,pd.DataFrame]], label: str) -> pd.DataFrame:
    if label == '6h':
        return aggregate_hashtags_6h(dfs)
    frames = []
    for name, df in tqdm(dfs, desc=f"Aggregating hashtags ({label})", unit="src"):
        if FAST_MODE and len(df) > SAMPLE_ROWS_PER_SOURCE:
            df = df.sample(SAMPLE_ROWS_PER_SOURCE, random_state=42)
        p = _prepare_text_df(df)
        if p is None or p.empty:
            continue
        pe = p[["ts","hashtags"]].explode("hashtags").dropna(subset=["hashtags"]) 
        if pe.empty:
            continue
        pe['bin'] = _assign_time_bin(pe['ts'], label)
        g = pe.groupby(['bin','hashtags'], as_index=False).size().rename(columns={'size':'count'})
        g = g.rename(columns={'hashtags':'feature'})
        g['category'] = g['feature'].map(categorize_feature)
        frames.append(g)
    if not frames:
        return pd.DataFrame(columns=["bin","feature","count","rolling_mean_24h","delta_vs_mean","category"])    
    allg = pd.concat(frames, ignore_index=True)
    allg = allg.groupby(["bin","feature","category"], as_index=False)['count'].sum()
    allg = _add_rolling_stats(allg, label)
    return allg

def aggregate_keywords_timeframe(dfs: List[Tuple[str,pd.DataFrame]], label: str) -> pd.DataFrame:
    if label == '6h':
        return aggregate_keywords_6h(dfs)
    frames = []
    kw_patterns = [(kw, re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE)) for kw in KEYWORDS]
    for name, df in tqdm(dfs, desc=f"Aggregating keywords ({label})", unit="src"):
        if FAST_MODE and len(df) > SAMPLE_ROWS_PER_SOURCE:
            df = df.sample(SAMPLE_ROWS_PER_SOURCE, random_state=42)
        p = _prepare_text_df(df)
        if p is None or p.empty:
            continue
        p['bin'] = _assign_time_bin(p['ts'], label)
        for kw, pat in kw_patterns:
            mask = p['text_clean'].str.contains(pat)
            if not mask.any():
                continue
            g = p.loc[mask].groupby('bin', as_index=False).size().rename(columns={'size':'count'})
            g['feature'] = kw
            g['category'] = categorize_feature(kw)
            frames.append(g)
    if not frames:
        return pd.DataFrame(columns=["bin","feature","count","rolling_mean_24h","delta_vs_mean","category"]) 
    allg = pd.concat(frames, ignore_index=True)
    allg = allg.groupby(["bin","feature","category"], as_index=False)['count'].sum()
    allg = _add_rolling_stats(allg, label)
    return allg

def aggregate_audio_timeframe(dfs: List[Tuple[str,pd.DataFrame]], label: str) -> pd.DataFrame:
    if label == '6h':
        return aggregate_audio_6h(dfs)
    frames: List[pd.DataFrame] = []
    for name, df in tqdm(dfs, desc=f"Aggregating audio ({label})", unit="src"):
        if FAST_MODE and len(df) > SAMPLE_ROWS_PER_SOURCE:
            df = df.sample(SAMPLE_ROWS_PER_SOURCE, random_state=42)
        ts_col = find_timestamp_column(df)
        audio_col = find_audio_column(df)
        if not ts_col or not audio_col:
            continue
        use = df[[ts_col, audio_col]].copy()
        use['ts'] = pd.to_datetime(use[ts_col], errors='coerce')
        use = use.dropna(subset=['ts', audio_col])
        if use.empty:
            continue
        use['bin'] = _assign_time_bin(use['ts'], label)
        g = use.groupby(['bin', audio_col], as_index=False).size().rename(columns={'size':'count'})
        g = g.rename(columns={audio_col:'feature'})
        g['category'] = 'Other'
        frames.append(g)
    if not frames:
        return pd.DataFrame(columns=["bin","feature","count","rolling_mean_24h","delta_vs_mean","category"]) 
    allg = pd.concat(frames, ignore_index=True)
    allg = allg.groupby(['bin','feature','category'], as_index=False)['count'].sum()
    allg = _add_rolling_stats(allg, label)
    return allg


def check_parquet_exists(name: str) -> bool:
    """Check if parquet file already exists with data."""
    out_path = PROC_DIR / name
    if out_path.exists():
        try:
            df = pd.read_parquet(out_path)
            if len(df) > 0:
                print(f"Skipping {name} - file already exists with {len(df)} rows")
                return True
        except Exception as e:
            print(f"[WARN] Failed to read existing {name}: {e}")
    return False


def write_parquet(df: pd.DataFrame, name: str, force: bool = False) -> Path:
    """Write parquet file, optionally skipping if it already exists with data."""
    out = PROC_DIR / name
    if not force and check_parquet_exists(name):
        return out
    df.to_parquet(out, index=False)
    print(f"Wrote {out}")
    return out


def write_phase3_report(emerging_df: pd.DataFrame, anomalies_df: pd.DataFrame, clusters_df: pd.DataFrame):
    """Write comprehensive Phase 3 report with advanced trend detection results."""
    path = INTERIM_DIR / 'phase3_advanced_trends_report.md'
    lines = ["# Phase 3 Advanced Trend Detection Report", ""]
    
    lines.append("## Executive Summary")
    lines.append(f"- Emerging terms detected: {len(emerging_df[emerging_df['is_emerging'] == True]) if not emerging_df.empty else 0}")
    lines.append(f"- Statistical anomalies found: {len(anomalies_df) if not anomalies_df.empty else 0}")
    lines.append(f"- Trend clusters identified: {len(clusters_df) if not clusters_df.empty else 0}")
    lines.append("")
    
    # Emerging Terms Analysis
    lines.append("## Emerging Terms (Keyword-Independent Detection)")
    if emerging_df.empty:
        lines.append("No emerging terms detected.")
    else:
        # Top emerging terms by velocity
        emerging_only = emerging_df[emerging_df['is_emerging'] == True]
        if not emerging_only.empty:
            latest_bin = emerging_only['bin'].max()
            top_emerging = emerging_only[emerging_only['bin'] == latest_bin].sort_values('velocity', ascending=False).head(20)
            
            lines.append(f"### Top Emerging Terms (Latest Period: {latest_bin})")
            try:
                cols = ['feature', 'count', 'growth_rate', 'velocity', 'category']
                cols = [c for c in cols if c in top_emerging.columns]
                lines.append(top_emerging[cols].to_markdown(index=False))
            except Exception as e:
                lines.append(f"(Could not render emerging terms table: {e})")
        
        # Growth rate analysis
        lines.append("### Growth Rate Distribution")
        if 'growth_rate' in emerging_df.columns:
            growth_stats = emerging_df['growth_rate'].describe()
            lines.append(f"- Mean growth rate: {growth_stats['mean']:.2f}")
            lines.append(f"- Max growth rate: {growth_stats['max']:.2f}")
            lines.append(f"- Terms with >5x growth: {len(emerging_df[emerging_df['growth_rate'] > 5])}")
    lines.append("")
    
    # Statistical Anomalies
    lines.append("## Statistical Anomalies")
    if anomalies_df.empty:
        lines.append("No statistical anomalies detected.")
    else:
        lines.append(f"Found {len(anomalies_df)} anomalous spikes in term frequency.")
        
        # Top anomalies by z-score
        top_anomalies = anomalies_df.sort_values('z_score', ascending=False).head(15)
        lines.append("### Top Anomalies by Z-Score")
        try:
            cols = ['feature', 'count', 'z_score', 'baseline_mean', 'velocity', 'category']
            cols = [c for c in cols if c in top_anomalies.columns]
            lines.append(top_anomalies[cols].to_markdown(index=False))
        except Exception as e:
            lines.append(f"(Could not render anomalies table: {e})")
    lines.append("")
    
    # Trend Clusters
    lines.append("## Trend Clusters")
    if clusters_df.empty:
        lines.append("No trend clusters identified.")
    else:
        lines.append(f"Identified {len(clusters_df)} trend clusters.")
        
        # Top clusters by average velocity
        top_clusters = clusters_df.sort_values('avg_velocity', ascending=False).head(10)
        lines.append("### Top Trend Clusters by Velocity")
        for _, cluster in top_clusters.iterrows():
            lines.append(f"**Cluster {cluster['cluster_id']}** (Velocity: {cluster['avg_velocity']:.2f})")
            lines.append(f"- Terms: {', '.join(cluster['terms'][:10])}{'...' if len(cluster['terms']) > 10 else ''}")
            lines.append(f"- Size: {cluster['cluster_size']} terms")
            lines.append("")
    
    # Category Analysis
    lines.append("## Category Analysis")
    category_data = []
    
    for df_name, df in [("Emerging", emerging_df), ("Anomalies", anomalies_df)]:
        if not df.empty and 'category' in df.columns:
            cat_counts = df['category'].value_counts()
            for cat, count in cat_counts.items():
                category_data.append({
                    'Type': df_name,
                    'Category': cat,
                    'Count': count
                })
    
    if category_data:
        cat_df = pd.DataFrame(category_data)
        try:
            lines.append(cat_df.to_markdown(index=False))
        except Exception as e:
            lines.append(f"(Could not render category analysis: {e})")
    else:
        lines.append("No category data available.")
    
    lines.append("")
    lines.append("## Methodology Notes")
    lines.append("- **Emerging Terms**: Detected using growth rate analysis and frequency thresholds")
    lines.append("- **Statistical Anomalies**: Identified using z-score analysis (threshold: 2.5)")
    lines.append("- **Trend Clusters**: Grouped using Jaccard similarity on term words")
    lines.append("- **Velocity**: Rate of change in term frequency over 24-hour windows")
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"Phase 3 report written to {path}")


def write_phase2_report(h_df: pd.DataFrame, k_df: pd.DataFrame, a_df: pd.DataFrame):
    path = INTERIM_DIR / 'phase2_features_report.md'
    lines = ["# Phase 2 Feature Engineering Report", ""]
    lines.append("## Summary (6h baseline)")
    lines.append(f"- Hashtag rows (6h): {len(h_df)}")
    lines.append(f"- Keyword rows (6h): {len(k_df)}")
    lines.append(f"- Audio rows (6h): {len(a_df)}")
    lines.append("")
    # Multi-frequency inventory
    inventory = []
    for p in PROC_DIR.glob('features_*.parquet'):
        if 'statistical_anomalies' in p.name or 'emerging_terms' in p.name:
            continue
        try:
            dfp = pd.read_parquet(p)
            label = p.stem.split('_')[-1]
            inventory.append({'file': p.name, 'rows': len(dfp), 'timeframe': label})
        except Exception:
            continue
    if inventory:
        lines.append("## Multi-Frequency Inventory")
        inv_df = pd.DataFrame(inventory).sort_values(['timeframe','file'])
        try:
            lines.append(inv_df.to_markdown(index=False))
        except Exception as e:
            lines.append(f"(Could not render inventory: {e})")
        lines.append("")
    for label, df in [("Hashtags (6h)", h_df), ("Keywords (6h)", k_df), ("Audio (6h)", a_df)]:
        lines.append(f"## {label}")
        if df.empty:
            lines.append("No data available.")
            lines.append("")
            continue
        latest_bin = df['bin'].max()
        latest = df[df['bin'] == latest_bin].sort_values('delta_vs_mean', ascending=False).head(15)
        try:
            lines.append(f"Latest bin: {latest_bin}")
            cols = [c for c in ["category", "feature", "count", "rolling_mean_24h", "delta_vs_mean"] if c in latest.columns]
            lines.append(latest[cols].to_markdown(index=False))
        except Exception as e:
            lines.append(f"(Could not render table: {e})")
        lines.append("")
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"Report written to {path}")

def build_trend_candidate_table():
    patterns = ["features_hashtags_*.parquet", "features_keywords_*.parquet"]
    rows = []
    for pat in patterns:
        for p in PROC_DIR.glob(pat):
            try:
                dfp = pd.read_parquet(p)
                if dfp.empty:
                    continue
                freq = p.stem.split('_')[-1]
                dfp = dfp.copy()
                if 'bin' in dfp.columns:
                    dfp['bin'] = pd.to_datetime(dfp['bin'])
                dfp = dfp.sort_values(['feature','bin'])
                if 'rolling_mean_24h' not in dfp.columns:
                    dfp['rolling_mean_24h'] = dfp.groupby('feature')['count'].transform(lambda s: s.rolling(window=3, min_periods=1).mean())
                dfp['delta_vs_mean'] = dfp['count'] - dfp['rolling_mean_24h']
                dfp['rate_of_change'] = dfp.groupby('feature')['count'].pct_change().replace([np.inf,-np.inf],0).fillna(0)
                dfp['frequency'] = freq
                rows.append(dfp[['bin','feature','count','rolling_mean_24h','delta_vs_mean','rate_of_change','category','frequency']])
            except Exception as e:
                print(f"[WARN] Skipping trend candidate build for {p.name}: {e}")
    if rows:
        master = pd.concat(rows, ignore_index=True)
        write_parquet(master, 'trend_candidates_master.parquet', force=True)
        print(f"[Phase3-FAST] Trend candidate master rows: {len(master)}")
    else:
        print("[Phase3-FAST] No trend candidates built.")


# -----------------------------
# Trend clustering (Phase 3 support) — simple Jaccard-based union-find
# -----------------------------

def identify_trend_clusters(emerging_df: pd.DataFrame, similarity_threshold: float = 0.6) -> pd.DataFrame:
    """Cluster terms using Jaccard similarity over token sets.

    Args:
        emerging_df: DataFrame with at least 'feature', optionally 'velocity'.
        similarity_threshold: Jaccard threshold to join clusters.
    Returns:
        DataFrame with cluster_id, terms list, cluster_size, avg_velocity.
    """
    if emerging_df is None or emerging_df.empty or 'feature' not in emerging_df.columns:
        return pd.DataFrame(columns=['cluster_id','terms','cluster_size','avg_velocity'])

    features = sorted(emerging_df['feature'].unique())
    token_map: Dict[str, Set[str]] = {f: set(re.findall(r"[a-zA-Z0-9]+", f.lower())) for f in features}

    # Union-Find structure
    parent = {f: f for f in features}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for f1, f2 in combinations(features, 2):
        t1, t2 = token_map[f1], token_map[f2]
        if not t1 or not t2:
            continue
        inter = len(t1 & t2)
        if inter == 0:
            continue
        jacc = inter / len(t1 | t2)
        if jacc >= similarity_threshold:
            union(f1, f2)

    # Group clusters
    clusters: Dict[str, List[str]] = defaultdict(list)
    for f in features:
        clusters[find(f)].append(f)

    records = []
    velocity_map = emerging_df.set_index('feature')['velocity'].to_dict() if 'velocity' in emerging_df.columns else {}
    for cid, terms in enumerate(clusters.values()):
        avg_vel = float(np.nanmean([velocity_map.get(t, np.nan) for t in terms])) if velocity_map else np.nan
        records.append({
            'cluster_id': cid,
            'terms': terms,
            'cluster_size': len(terms),
            'avg_velocity': avg_vel
        })
    return pd.DataFrame(records).sort_values('cluster_size', ascending=False).reset_index(drop=True)


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

    # Extended multi-timeframe aggregations
    print("[Phase2] Multi-timeframe aggregations starting (all sources)...")
    all_sources = list(samples.items())
    for label in TIMEFRAME_LABELS:
        if label == '6h':
            continue
        try:
            h_name = f'features_hashtags_{label}.parquet'
            k_name = f'features_keywords_{label}.parquet'
            a_name = f'features_audio_{label}.parquet'
            if not check_parquet_exists(h_name):
                h_tf = aggregate_hashtags_timeframe(all_sources, label)
                if not h_tf.empty:
                    write_parquet(h_tf, h_name)
            if not check_parquet_exists(k_name):
                k_tf = aggregate_keywords_timeframe(all_sources, label)
                if not k_tf.empty:
                    write_parquet(k_tf, k_name)
            if not check_parquet_exists(a_name):
                a_tf = aggregate_audio_timeframe(all_sources, label)
                if not a_tf.empty:
                    write_parquet(a_tf, a_name)
        except Exception as e:
            print(f"[WARN] Failed timeframe aggregation {label}: {e}")
    print("[Phase2] Multi-timeframe aggregations completed (all sources).")

    write_phase2_report(ts_hashtags, ts_keywords, ts_audio)
    
    # Phase 3 FAST simplified trend detection
    print("\n[Phase3-FAST] Starting simplified trend detection...")
    ts_emerging = aggregate_emerging_terms_6h(all_sources, min_frequency=5, min_growth_rate=2.5)
    if FAST_MODE and len(ts_emerging) > MAX_FEATURES_EMERGING:
        ts_emerging = ts_emerging.sort_values('count', ascending=False).head(MAX_FEATURES_EMERGING)
    all_traditional = pd.concat([ts_hashtags, ts_keywords], ignore_index=True)
    if FAST_MODE:
        # keep top features by total count
        top_feats = all_traditional.groupby('feature')['count'].sum().sort_values(ascending=False).head(MAX_FEATURES_ANOMALY).index
        all_traditional = all_traditional[all_traditional['feature'].isin(top_feats)]
    ts_anomalies = detect_statistical_anomalies(all_traditional, z_threshold=3.0 if FAST_MODE else 2.5)
    # Simple cluster grouping by first token prefix
    def simple_cluster(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=['cluster_id','terms','cluster_size','avg_velocity'])
        groups = defaultdict(list)
        for term in df['feature'].unique():
            key = term.split()[0][:5].lower()
            groups[key].append(term)
        rows = []
        cid = 0
        for k, feats in groups.items():
            rows.append({'cluster_id': cid, 'terms': feats, 'cluster_size': len(feats), 'avg_velocity': float(df[df['feature'].isin(feats)]['velocity'].mean() if 'velocity' in df.columns else 0.0)})
            cid += 1
        return pd.DataFrame(rows)
    ts_clusters = simple_cluster(ts_emerging)
    if not check_parquet_exists('features_emerging_terms_6h.parquet'):
        write_parquet(ts_emerging, 'features_emerging_terms_6h.parquet')
    if not ts_anomalies.empty and not check_parquet_exists('features_statistical_anomalies.parquet'):
        write_parquet(ts_anomalies, 'features_statistical_anomalies.parquet')
    if not ts_clusters.empty and not check_parquet_exists('trend_clusters.parquet'):
        write_parquet(ts_clusters, 'trend_clusters.parquet')
    write_phase3_report(ts_emerging, ts_anomalies, ts_clusters)
    print("[Phase3-FAST] Simplified trend detection completed.")
    print("Summary:")
    print(f"  - Emerging terms (kept): {len(ts_emerging[ts_emerging['is_emerging'] == True]) if not ts_emerging.empty else 0}")
    print(f"  - Statistical anomalies: {len(ts_anomalies)}")
    print(f"  - Simple clusters: {len(ts_clusters)}")
    build_trend_candidate_table()


if __name__ == '__main__':
    main()
