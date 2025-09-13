import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import json
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import random

# Load data here
# -----------------------
# Real Data from Enhanced Modeling Results
# -----------------------

st.set_page_config(layout="wide", page_title="L'Oréal TrendSpotter Dashboard")

# Custom CSS
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
            padding-left: 10rem;
            padding-right: 10rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

def load_real_data():
    """Load real data from enhanced modeling results."""
    try:
        # Load enhanced modeling results
        results_file = Path("../models/enhanced_modeling_results.json")
        if not results_file.exists():
            return None
        
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Extract trending terms data
        trends_data = []
        
        # Get overall category terms
        overall_category_terms = results.get('overall_category_terms', {})
        
        for category, terms in overall_category_terms.items():
            for term_data in terms:
                trends_data.append({
                    'trend_id': len(trends_data) + 1,
                    'name': term_data['term'],
                    'trend_score': int(term_data['tfidf_score'] * 1000),  # Scale up for display
                    'platform': 'Multi-Platform',  # Real data doesn't have platform breakdown
                    'category': category.title(),
                    'demographic': 'All Ages',  # Real data doesn't have demographic breakdown
                    'sentiment': 'positive',  # Assuming trending terms are positive
                    'sentiment_score': min(0.95, 0.5 + term_data['tfidf_score']),
                    'keywords': term_data['term'],
                    'tfidf_score': term_data['tfidf_score']
                })
        
        return pd.DataFrame(trends_data)
        
    except Exception as e:
        st.error(f"Error loading real data: {e}")
        return None

def load_decay_analysis_data():
    """Load real decay analysis data."""
    try:
        # Load both real_term_decay_analysis_results.json and term_decay_analysis_results.json
        decay_file = Path("../models/real_term_decay_analysis_results.json")
        term_decay_file = Path("../models/term_decay_analysis_results.json")
        term_decay_top10_file = Path("term_decay_analysis_results_top10.json")
        
        decay_results = None
        term_decay_results = None
        
        if decay_file.exists():
            with open(decay_file, 'r') as f:
                decay_results = json.load(f)
        
        # For search functionality, prefer the full file over top10 to get all terms
        if term_decay_file.exists():
            with open(term_decay_file, 'r') as f:
                term_decay_results = json.load(f)
        elif term_decay_top10_file.exists():
            with open(term_decay_top10_file, 'r') as f:
                term_decay_results = json.load(f)
        
        return {
            'real_decay': decay_results,
            'term_decay': term_decay_results
        }
    except Exception as e:
        st.error(f"Error loading decay analysis: {e}")
        return None

def get_all_available_terms():
    """Extract all terms from modeling results and decay analysis for search functionality."""
    all_terms = set()
    
    # Add terms from the current dataframe (top terms from enhanced_modeling_results.json)
    if 'df_trends' in st.session_state:
        df_terms = st.session_state.df_trends['name'].tolist()
        all_terms.update(df_terms)
    
    # Add terms from enhanced modeling results (all categories, all time windows)
    if 'results_data' in st.session_state and st.session_state.results_data:
        results_data = st.session_state.results_data
        
        # Extract from overall_category_terms
        overall_category_terms = results_data.get('overall_category_terms', {})
        for category, terms in overall_category_terms.items():
            for term_data in terms:
                all_terms.add(term_data['term'])
        
        # Extract from temporal_trends
        temporal_trends = results_data.get('temporal_trends', {})
        for category, category_data in temporal_trends.items():
            trending_terms_by_window = category_data.get('trending_terms_by_window', {})
            for time_window, terms in trending_terms_by_window.items():
                if terms:
                    for term_data in terms:
                        all_terms.add(term_data['term'])
    
    # Add terms from decay analysis
    decay_data = st.session_state.get('decay_data', {})
    if decay_data:
        term_decay_analysis = decay_data.get('term_decay', {})
        if not term_decay_analysis and 'term_decay_analysis' in decay_data:
            term_decay_analysis = decay_data['term_decay_analysis']
        
        if term_decay_analysis:
            for category_name, category_terms in term_decay_analysis.items():
                if isinstance(category_terms, dict):
                    for term_name in category_terms.keys():
                        all_terms.add(term_name)
    
    # Convert to sorted list for better UX
    return sorted(list(all_terms))

# Load real data or fallback to sample data
if 'df_trends' not in st.session_state:
    real_data = load_real_data()
    
    if real_data is not None:
        st.session_state.df_trends = real_data
        st.session_state.data_source = "real"
        
        # Also load the full results data for temporal analysis
        try:
            results_file = Path("../models/enhanced_modeling_results.json")
            if results_file.exists():
                with open(results_file, 'r') as f:
                    st.session_state.results_data = json.load(f)
        except Exception as e:
            st.session_state.results_data = None
    else:
        # Fallback to sample data if real data can't be loaded
        st.warning("Could not load real data. Using sample data for demonstration.")
        
        base_trends = [
            ("#GlowSkin", "Skincare"),
            ("#SunProtection", "Makeup"),
            ("Viral Audio 123", "Lifestyle"),
            ("#MakeupTutorial", "Makeup"),
            ("#HairCareHack", "Hair"),
            ("Viral Dance Beat", "Audio"),
            ("#EcoBeauty", "Skincare"),
            ("ASMR Lipstick Test", "Makeup"),
            ("#NoFilterChallenge", "Lifestyle"),
            ("Viral Audio XYZ", "Audio")
        ]

        platforms = ["TikTok", "Instagram", "YouTube", "Twitter"]
        demographics = ["Gen Z", "Millennials", "Gen Alpha"]
        sentiments = ["positive", "neutral", "negative"]

        # Expand to 100 rows
        rows = []
        for i in range(1, 101):
            name, category = base_trends[i % len(base_trends)]
            rows.append({
                "trend_id": i,
                "name": f"{name}_{i}",
                "trend_score": random.randint(60, 100),
                "platform": random.choice(platforms),
                "category": category,
                "demographic": random.choice(demographics),
                "sentiment": random.choice(sentiments),
                "sentiment_score": round(random.uniform(0.2, 0.95), 2),
                "keywords": f"keywords for {name.lower()}_{i}"
            })

        st.session_state.df_trends = pd.DataFrame(rows)
        st.session_state.data_source = "sample"

    # Clean up data
    st.session_state.df_trends["platform"] = st.session_state.df_trends["platform"].str.strip().str.title()
    st.session_state.df_trends["category"] = st.session_state.df_trends["category"].str.strip().str.title()
    st.session_state.df_trends["demographic"] = st.session_state.df_trends["demographic"].str.strip()

# Load decay analysis data
if 'decay_data' not in st.session_state:
    st.session_state.decay_data = load_decay_analysis_data()

# Load all available terms for search
if 'all_available_terms' not in st.session_state:
    st.session_state.all_available_terms = get_all_available_terms()

def get_term_data_from_decay_analysis(term_name, decay_data):
    """Get term data from decay analysis for a specific term."""
    if not decay_data:
        return None
    
    # Clean term name for matching
    clean_term_name = term_name.lower().replace('#', '').replace('_', ' ').strip()
    
    # Check term_decay_analysis structure
    term_decay_analysis = decay_data.get('term_decay', {})
    if not term_decay_analysis and 'term_decay_analysis' in decay_data:
        term_decay_analysis = decay_data['term_decay_analysis']
    
    if term_decay_analysis:
        # Search through all categories for the term
        for category_name, category_terms in term_decay_analysis.items():
            if isinstance(category_terms, dict):
                # Try exact match first
                if clean_term_name in category_terms:
                    term_data = category_terms[clean_term_name]
                    if isinstance(term_data, dict):
                        return term_data
                
                # Try partial match
                for k, v in category_terms.items():
                    if isinstance(v, dict):
                        if (k in clean_term_name) or (clean_term_name in k):
                            return v
    
    return None

def load_temporal_data_for_term(term_name, results_data):
    """Load real temporal data from enhanced modeling results for a specific term."""
    if not results_data:
        return None
    
    try:
        # Clean term name for better matching
        clean_term_name = term_name.lower().replace('#', '').replace('_', ' ').strip()
        
        # Try all categories, not just skincare
        temporal_trends = results_data.get('temporal_trends', {})
        
        best_match_data = None
        best_match_score = 0
        
        for category, category_data in temporal_trends.items():
            trending_terms_by_window = category_data.get('trending_terms_by_window', {})
            
            timestamps = []
            scores = []
            
            for time_window, terms in trending_terms_by_window.items():
                if not terms:
                    continue
                    
                try:
                    start_date = time_window.split('/')[0]
                    timestamp = pd.to_datetime(start_date)
                    
                    # Look for matching term in this time window
                    term_score = 0
                    found_match = False
                    
                    for term_data in terms:
                        term_clean = term_data['term'].lower().replace('#', '').replace('_', ' ').strip()
                        if clean_term_name in term_clean or term_clean in clean_term_name:
                            term_score = term_data['tfidf_score']
                            found_match = True
                            break
                    
                    # If no exact match, check for partial matches with higher threshold
                    if not found_match:
                        for term_data in terms:
                            term_clean = term_data['term'].lower().replace('#', '').replace('_', ' ').strip()
                            # Check for partial word matches
                            if any(word in term_clean for word in clean_term_name.split() if len(word) > 2):
                                term_score = term_data['tfidf_score'] * 0.5  # Lower confidence for partial matches
                                found_match = True
                                break
                    
                    if found_match:
                        timestamps.append(timestamp)
                        scores.append(term_score * 1000)  # Scale for visualization
                        
                except Exception:
                    continue
            
            # Check if this category gave us better data
            if timestamps and scores:
                df = pd.DataFrame({
                    'timestamp': timestamps,
                    'tfidf_score': scores
                }).sort_values('timestamp')
                
                # Score this match based on data points and relevance
                match_score = len(df) * (sum(scores) / len(scores)) if scores else 0
                
                if match_score > best_match_score:
                    best_match_score = match_score
                    best_match_data = df
        
        return best_match_data
        
    except Exception as e:
        print(f"Error loading temporal data: {e}")
        return None

def create_synthetic_historical_data_for_term(term_name, term_data=None):
    """Create synthetic historical data based on term characteristics."""
    if term_data:
        max_tfidf = term_data.get('max_tfidf', 0.1)
        avg_growth_rate = term_data.get('avg_growth_rate', 0.0)
        appearances = term_data.get('appearances', 50)
        total_windows = term_data.get('total_windows', 290)
    else:
        # Default values
        max_tfidf = 0.1
        avg_growth_rate = 0.0
        appearances = 50
        total_windows = 290
    
    # Create timestamps spanning approximately 2 years
    start_date = datetime(2019, 1, 1)
    end_date = datetime(2021, 6, 30)
    
    # Generate timestamps
    np.random.seed(hash(term_name) % 2**32)  # Consistent but unique per term
    all_timestamps = pd.date_range(start_date, end_date, freq='W')
    
    # Select timestamps where term appeared
    appearance_indices = np.random.choice(len(all_timestamps), min(appearances, len(all_timestamps)), replace=False)
    appearance_indices = np.sort(appearance_indices)
    
    timestamps = [all_timestamps[i] for i in appearance_indices]
    
    # Generate TF-IDF scores with trend
    scores = []
    base_score = max_tfidf * 1000  # Scale for visualization
    
    for i, timestamp in enumerate(timestamps):
        # Apply trend over time
        progress = i / len(timestamps) if len(timestamps) > 0 else 0
        trend_factor = 1 + (avg_growth_rate * progress * 10)  # Amplify for visibility
        
        # Add some randomness but maintain overall trend
        noise = np.random.normal(0, base_score * 0.1)
        score = max(0, base_score * trend_factor + noise)
        scores.append(score)
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'tfidf_score': scores
    })

def predict_future_trend_for_term(historical_data, term_data=None, months=3):
    """Predict future trend for the next 3 months based on historical data and decay analysis."""
    if historical_data is None or historical_data.empty:
        return pd.DataFrame()
    
    # Get the last data point
    last_timestamp = historical_data['timestamp'].max()
    last_score = historical_data['tfidf_score'].iloc[-1]
    
    # Generate future timestamps (weekly intervals)
    future_timestamps = pd.date_range(
        start=last_timestamp + timedelta(weeks=1),
        periods=12,  # 3 months of weekly data
        freq='W'
    )
    
    # Use the actual decay characteristics if available
    if term_data:
        avg_growth_rate = term_data.get('avg_growth_rate', 0.0)
        trend_state = term_data.get('trend_state', 'Stable')
    else:
        avg_growth_rate = 0.02  # Default mild growth
        trend_state = 'Stable'
    
    # Generate predicted scores
    future_scores = []
    current_score = last_score
    
    for i, timestamp in enumerate(future_timestamps):
        # Apply the average growth rate
        current_score *= (1 + avg_growth_rate)
        
        # Add some uncertainty/noise
        noise = np.random.normal(0, current_score * 0.05)
        predicted_score = max(0, current_score + noise)
        future_scores.append(predicted_score)
    
    return pd.DataFrame({
        'timestamp': future_timestamps,
        'tfidf_score': future_scores
    })

def create_real_temporal_data(trend_name, results_data):
    """Create comprehensive temporal trend data for any term."""
    if not results_data:
        return create_fallback_temporal_data(trend_name)
    
    # Try to load real temporal data first
    historical_data = load_temporal_data_for_term(trend_name, results_data)
    
    # Get term-specific data from decay analysis
    decay_data = st.session_state.get('decay_data', {})
    term_data = get_term_data_from_decay_analysis(trend_name, decay_data)
    
    # If no real data, create synthetic data based on term characteristics
    if historical_data is None or historical_data.empty:
        historical_data = create_synthetic_historical_data_for_term(trend_name, term_data)
        data_type = "synthetic"
    else:
        data_type = "real"
    
    # Generate future predictions
    future_data = predict_future_trend_for_term(historical_data, term_data)
    
    # Add metadata for debugging
    historical_data['data_type'] = data_type
    if term_data:
        historical_data['trend_state'] = term_data.get('trend_state', 'Unknown')
        historical_data['avg_growth_rate'] = term_data.get('avg_growth_rate', 0.0)
    
    return historical_data

def create_fallback_temporal_data(trend_name):
    """Create fallback temporal data when no real data is available."""
    base_date = pd.Timestamp('2024-01-01')
    periods = 15
    timestamps = pd.date_range(base_date, periods=periods, freq='W')
    
    # Create varied patterns based on term name hash for consistency
    term_hash = hash(trend_name) % 1000
    base_score = 10 + (term_hash % 50)  # Base score between 10-60
    growth_factor = 0.98 + (term_hash % 40) / 1000  # Growth factor between 0.98-1.02
    
    scores = []
    current_score = base_score
    for i in range(periods):
        current_score *= growth_factor
        noise = random.randint(-5, 10)
        scores.append(max(1, int(current_score + noise)))
    
    df = pd.DataFrame({'timestamp': timestamps, 'count': scores})
    df['data_type'] = 'fallback'
    df['trend_state'] = 'Simulated'
    df['avg_growth_rate'] = growth_factor - 1
    
    return df

#timeseries
if 'ts' not in st.session_state:
    st.session_state.ts = pd.DataFrame({
        "timestamp": pd.date_range("2025-09-01", periods=10, freq="D"),
        "count": [5, 10, 15, 40, 60, 90, 120, 180, 150, 130]
    })

if 'applied_filters' not in st.session_state:
    st.session_state.applied_filters = {
        'platform': [],
        'category': [],
        'demographic': []
    }

if 'show_trend_details' not in st.session_state:
    st.session_state.show_trend_details = False

if 'selected_trend' not in st.session_state:
    st.session_state.selected_trend = None

# take the dataframe from previous load instead of fresh load everytime
df_trends = st.session_state.df_trends
ts = st.session_state.ts

#Dashboard View

st.title("L'Oréal TrendSpotter Dashboard")

# Show data source
if st.session_state.get('data_source') == 'real':
    st.success("📊 Using Real L'Oréal Datathon 2025 Model Results")
    
    # Debug info for temporal data
    if st.session_state.get('results_data'):
        temporal_data = st.session_state.results_data.get('temporal_trends', {}).get('skincare', {})
        window_count = len(temporal_data.get('trending_terms_by_window', {}))
        st.info(f"🕒 Temporal data loaded: {window_count} time windows available")
    else:
        st.warning("⚠️ Temporal data not loaded")
    
    # Debug info for decay data
    if st.session_state.get('decay_data'):
        decay_data = st.session_state.decay_data
        term_decay_analysis = decay_data.get('term_decay', {})
        if not term_decay_analysis and 'term_decay_analysis' in decay_data:
            term_decay_analysis = decay_data['term_decay_analysis']
        
        if term_decay_analysis:
            total_decay_terms = sum(len(cat_terms) for cat_terms in term_decay_analysis.values() if isinstance(cat_terms, dict))
            st.info(f"📉 Decay analysis loaded: {total_decay_terms} terms across {len(term_decay_analysis)} categories")
        else:
            st.warning("⚠️ Decay analysis data not loaded properly")
    else:
        st.warning("⚠️ Decay analysis not loaded")
else:
    st.info("📊 Using Sample Data (Real data not available)")

leaderboard, gap, filters = st.columns([6.5, 0.25, 2.75])

with filters:
    st.subheader("Filters")

    with st.form("filter_form"):
        st.markdown("**Select your filters:**")
        
        platform = st.multiselect(
            "Platform", 
            df_trends["platform"].unique(),
            default=st.session_state.applied_filters['platform']
        )
        
        category = st.multiselect(
            "Category", 
            df_trends["category"].unique(),
            default=st.session_state.applied_filters['category']
        )
        
        demographic = st.multiselect(
            "Demographic", 
            df_trends["demographic"].unique(),
            default=st.session_state.applied_filters['demographic']
        )
        
        # Form buttons
        col1, col2 = st.columns(2)
        with col1:
            apply_filters = st.form_submit_button("Apply Filters", type="primary", use_container_width=True)
        with col2:
            clear_filters = st.form_submit_button("Clear All", use_container_width=True)
    
    if apply_filters:
        st.session_state.applied_filters = {
            'platform': platform,
            'category': category,
            'demographic': demographic
        }
        st.success("Filters applied")
    
    if clear_filters:
        st.session_state.applied_filters = {
            'platform': [],
            'category': [],
            'demographic': []
        }
        st.success("Filters cleared")
        st.rerun() #reset state

# Apply filters
filtered = df_trends.copy()
if platform:
    filtered = filtered[filtered["platform"].isin(platform)]
if category:
    filtered = filtered[filtered["category"].isin(category)]
if demographic:
    filtered = filtered[filtered["demographic"].isin(demographic)]

# Data Summary Section
if st.session_state.get('data_source') == 'real':
    st.subheader("📈 Data Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Trends", len(filtered))
    with col2:
        st.metric("Categories", filtered['category'].nunique())
    with col3:
        avg_score = filtered['trend_score'].mean()
        st.metric("Avg Trend Score", f"{avg_score:.0f}")
    with col4:
        top_category = filtered['category'].value_counts().index[0]
        st.metric("Top Category", top_category)

with leaderboard:
    # leaderboard
    st.subheader("🔥 Top 10 Trends")
    top20 = (
        filtered[["name", "platform", "category", "trend_score"]]
        .sort_values("trend_score", ascending=False)
        .head(10)
        .rename(columns={
            "name": "Trend Name / Hashtag",
            "platform": "Platform",
            "category": "Category",
            "trend_score": "Trend Score"
        }).reset_index(drop=True) 
    )
    st.dataframe(top20)

# Main search/selection interface
# Initialize session state for the selected trend
if 'selected_trend' not in st.session_state:
    st.session_state.selected_trend = None

# Main search interface
st.subheader("Trend Search")

# Use all available terms from modeling and decay analysis, not just filtered trends
all_terms = st.session_state.get('all_available_terms', [])

# If we have comprehensive term data, use it; otherwise fallback to filtered data
if all_terms and len(all_terms) > len(filtered["name"].unique()):
    options = all_terms
    total_terms = len(all_terms)
    st.caption(f"🔍 Search from {total_terms} total terms across all modeling and decay analysis data")
else:
    options = filtered["name"].unique()
    total_terms = len(options)
    st.caption(f"🔍 Search from {total_terms} terms (filtered view)")

label = f"Start typing to search from {total_terms} available trends..."

trend_choice = st.selectbox(label, options)

# Define the dialog function
@st.dialog("Trend Analysis", width = "large")
def show_trend_details(trend_name):
    st.title(f"📊 {trend_name}")
    
    # Get trend data
    trend_row = df_trends[df_trends['name'] == trend_name].iloc[0] if len(df_trends[df_trends['name'] == trend_name]) > 0 else None
    
    # Get term-specific data from decay analysis
    decay_data = st.session_state.get('decay_data', {})
    term_data = get_term_data_from_decay_analysis(trend_name, decay_data)
    
    # Create row with time plot and insights
    col1, col2, col3 = st.columns([6, 0.5, 3.5])
    
    with col1:
        # Use enhanced temporal data logic
        if st.session_state.get('data_source') == 'real' and st.session_state.get('results_data'):
            # Load historical data using enhanced logic
            historical_data = load_temporal_data_for_term(trend_name, st.session_state.results_data)
            
            # If no real data found, create synthetic data based on term characteristics
            if historical_data is None or historical_data.empty:
                historical_data = create_synthetic_historical_data_for_term(trend_name, term_data)
                data_type = "synthetic"
                st.caption(f"📊 Synthetic data based on term characteristics")
            else:
                data_type = "real"
                date_range = f"{historical_data['timestamp'].min().strftime('%Y-%m-%d')} to {historical_data['timestamp'].max().strftime('%Y-%m-%d')}"
                st.caption(f"📊 Real temporal data from {len(historical_data)} time windows ({date_range})")
            
            # Create enhanced historical visualization
            fig_hist = go.Figure()
            fig_hist.add_trace(
                go.Scatter(
                    x=historical_data['timestamp'],
                    y=historical_data['tfidf_score'],
                    mode='lines+markers',
                    name='Historical Trend',
                    line=dict(color='#2E86AB', width=3),
                    marker=dict(size=6)
                )
            )
            
            # Add term metadata to title if available
            title_suffix = ""
            if term_data:
                trend_state = term_data.get('trend_state', 'Unknown')
                avg_growth_rate = term_data.get('avg_growth_rate', 0)
                title_suffix = f" - {trend_state} (Rate: {avg_growth_rate:.3f})"
            
            fig_hist.update_layout(
                title=f"Historical Trend: {trend_name}{title_suffix}",
                xaxis_title="Date",
                yaxis_title="TF-IDF Score (×1000)" if data_type == "real" else "Engagement Score",
                template="plotly_white",
                height=400
            )
            
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Generate predictions using enhanced logic
            future_data = predict_future_trend_for_term(historical_data, term_data)
            
            if not future_data.empty:
                # Create prediction visualization
                fig_pred = go.Figure()
                
                # Add last few historical points for context
                context_data = historical_data.tail(5)
                fig_pred.add_trace(
                    go.Scatter(
                        x=context_data['timestamp'],
                        y=context_data['tfidf_score'],
                        mode='lines+markers',
                        name='Historical Context',
                        line=dict(color='#2E86AB', width=2),
                        marker=dict(size=4)
                    )
                )
                
                # Add predictions
                fig_pred.add_trace(
                    go.Scatter(
                        x=future_data['timestamp'],
                        y=future_data['tfidf_score'],
                        mode='lines+markers',
                        name='3-Month Prediction',
                        line=dict(color='#F24236', dash='dash', width=3),
                        marker=dict(size=6, symbol='triangle-up')
                    )
                )
                
                # Add separation line
                fig_pred.add_shape(
                    type="line",
                    x0=future_data['timestamp'].min(),
                    x1=future_data['timestamp'].min(),
                    y0=0, y1=1,
                    yref="paper",
                    line=dict(color='gray', dash='dot', width=2)
                )
                
                fig_pred.update_layout(
                    title=f"3-Month Prediction: {trend_name}",
                    xaxis_title="Date",
                    yaxis_title="Predicted Score",
                    template="plotly_white",
                    height=400
                )
                
                st.plotly_chart(fig_pred, use_container_width=True)
                
                # Show prediction summary
                total_change = future_data['tfidf_score'].iloc[-1] - future_data['tfidf_score'].iloc[0]
                monthly_change = total_change / 3
                
                if total_change > 0:
                    trend_direction = "📈 Growth"
                    change_color = "green"
                elif total_change < 0:
                    trend_direction = "📉 Decline"  
                    change_color = "red"
                else:
                    trend_direction = "📊 Stable"
                    change_color = "blue"
                
                st.markdown(f"**Prediction Summary:** {trend_direction}")
                st.markdown(f"- **3-Month Change:** {total_change:.2f} points")
                st.markdown(f"- **Monthly Rate:** {monthly_change:.2f} points/month")
            
        else:
            # Fallback to sample data
            historical_data = create_fallback_temporal_data(trend_name)
            fig = px.line(historical_data, x="timestamp", y="count", 
                         title=f"Simulated Trend: {trend_name}", 
                         labels={"timestamp": "Date", "count": "Trend Score"})
            st.caption("📊 Sample data (real data not available)")
            st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        st.subheader("🔍 Enhanced Insights")
        
        if st.session_state.get('data_source') == 'real':
            # Try to get category information from multiple sources
            category = None
            tfidf_score = None
            trend_score = None
            
            # First try df_trends if term exists there
            if trend_row is not None:
                category = trend_row['category']
                tfidf_score = trend_row.get('tfidf_score', 0)
                trend_score = trend_row['trend_score']
            else:
                # Try to find category from enhanced modeling results
                results_data = st.session_state.get('results_data', {})
                if results_data:
                    overall_category_terms = results_data.get('overall_category_terms', {})
                    for cat_name, terms in overall_category_terms.items():
                        for term_data_item in terms:
                            if term_data_item['term'].lower() == trend_name.lower():
                                category = cat_name.title()
                                tfidf_score = term_data_item['tfidf_score']
                                trend_score = int(tfidf_score * 1000)
                                break
                        if category:
                            break
                
                # If still no category, try to infer from decay analysis
                if not category and term_data:
                    # Find which category this term belongs to in decay analysis
                    decay_data = st.session_state.get('decay_data', {})
                    term_decay_analysis = decay_data.get('term_decay', {})
                    if not term_decay_analysis and 'term_decay_analysis' in decay_data:
                        term_decay_analysis = decay_data['term_decay_analysis']
                    
                    clean_term_name = trend_name.lower().replace('#', '').replace('_', ' ').strip()
                    for cat_name, cat_terms in term_decay_analysis.items():
                        if isinstance(cat_terms, dict) and clean_term_name in cat_terms:
                            category = cat_name.title()
                            break
            
            # Display available information
            if category:
                st.write(f"**Category:** {category}")
            else:
                st.write("**Category:** Unknown")
                
            if tfidf_score is not None:
                st.write(f"**TF-IDF Score:** {tfidf_score:.4f}")
            if trend_score is not None:
                st.write(f"**Trend Score:** {trend_score}")
            
            # Add term-specific decay analysis insights
            if term_data:
                st.write("---")
                st.write("**Decay Analysis:**")
                trend_state = term_data.get('trend_state', 'Unknown')
                avg_growth_rate = term_data.get('avg_growth_rate', 0)
                appearances = term_data.get('appearances', 0)
                total_windows = term_data.get('total_windows', 0)
                decay_confidence = term_data.get('decay_confidence', 0)
                
                # Trend state with emoji
                state_emoji = {
                    'Accelerating': '🚀',
                    'Declining': '📉', 
                    'Stable': '📊',
                    'Decaying': '⚰️'
                }.get(trend_state, '❓')
                
                st.write(f"- **Trend State:** {state_emoji} {trend_state}")
                st.write(f"- **Growth Rate:** {avg_growth_rate:.4f}")
                
                # Prevent division by zero
                if total_windows > 0:
                    coverage_pct = (appearances/total_windows*100)
                    st.write(f"- **Data Coverage:** {appearances}/{total_windows} windows ({coverage_pct:.1f}%)")
                else:
                    st.write(f"- **Data Coverage:** {appearances} windows (unknown total)")
                    
                st.write(f"- **Decay Confidence:** {decay_confidence:.3f}")
                
                if trend_state == 'Accelerating':
                    st.success(f"🚀 **TRENDING UP!** This term is gaining momentum!")
                elif trend_state == 'Declining':
                    st.warning(f"📉 **Declining trend** - consider pivot strategies")
                elif trend_state == 'Decaying':
                    st.error(f"⚰️ **Strong decline** detected - immediate action needed")
            else:
                st.info("⚠️ No specific decay analysis data found for this term")
            
            st.write("---")
            st.write("**Strategic Insights:**")
            if category:
                st.write(f"- '{trend_name}' shows relevance in {category.lower()} discussions")
            else:
                st.write(f"- '{trend_name}' found in comprehensive modeling data")
            st.write(f"- Based on **real social media data** analysis")
            st.write(f"- Identified through **advanced NLP modeling**")
            
            # Add data source information
            if historical_data is not None and not historical_data.empty:
                data_points = len(historical_data)
                score_range = f"{historical_data['tfidf_score'].min():.2f} - {historical_data['tfidf_score'].max():.2f}"
                st.write(f"- **Data Points:** {data_points} time windows")
                st.write(f"- **Score Range:** {score_range}")
        
        else:
            # Sample data insights
            st.write(f"- {trend_name} is currently **growing 25% week-over-week**.")
            st.write(f"- Popular among **Millennials** with **80% positive sentiment**.")
            st.write(f"- *Note: Using sample data for demonstration*")


# Button to trigger the dialog
if st.button("View Details", disabled=not trend_choice):
    # call function here to retrieve the trend data. smth like retrieve_trend_data(trend_choice)
    # for now the name is provided as input
    # smth like 
    show_trend_details(trend_choice)