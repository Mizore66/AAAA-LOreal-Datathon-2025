import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import json
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
        results_file = Path("../data/interim/enhanced_modeling_results.json")
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
        decay_file = Path("../data/interim/real_term_decay_analysis_results.json")
        if not decay_file.exists():
            return None
            
        with open(decay_file, 'r') as f:
            decay_results = json.load(f)
        
        return decay_results
        
    except Exception as e:
        st.error(f"Error loading decay analysis: {e}")
        return None

# Load real data or fallback to sample data
if 'df_trends' not in st.session_state:
    real_data = load_real_data()
    
    if real_data is not None:
        st.session_state.df_trends = real_data
        st.session_state.data_source = "real"
        
        # Also load the full results data for temporal analysis
        try:
            results_file = Path("../data/interim/enhanced_modeling_results.json")
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

def create_real_temporal_data(trend_name, results_data):
    """Create real temporal trend data from enhanced modeling results."""
    try:
        # Extract temporal trends for skincare category (most data available)
        temporal_trends = results_data.get('temporal_trends', {}).get('skincare', {})
        trending_terms_by_window = temporal_trends.get('trending_terms_by_window', {})
        
        # Create time series data
        timestamps = []
        scores = []
        
        # Clean trend name for better matching
        clean_trend_name = trend_name.lower().replace('#', '').replace('_', ' ').strip()
        
        for time_window, terms in trending_terms_by_window.items():
            if not terms:  # Skip empty windows
                continue
                
            # Parse time window (format: "2020-01-06/2020-01-12")
            try:
                start_date = time_window.split('/')[0]
                timestamp = pd.to_datetime(start_date)
                timestamps.append(timestamp)
                
                # Find matching term or use average score
                term_score = 0
                found_match = False
                
                # Try exact match first
                for term_data in terms:
                    term_clean = term_data['term'].lower().replace('#', '').replace('_', ' ').strip()
                    if clean_trend_name in term_clean or term_clean in clean_trend_name:
                        term_score = term_data['tfidf_score'] * 1000  # Scale for display
                        found_match = True
                        break
                
                # If no match found, use weighted average of top terms
                if not found_match and terms:
                    # Weight higher-ranked terms more heavily
                    weighted_sum = 0
                    weight_total = 0
                    for i, term_data in enumerate(terms[:5]):  # Top 5 terms
                        weight = 1.0 / (i + 1)  # Higher weight for higher rank
                        weighted_sum += term_data['tfidf_score'] * weight
                        weight_total += weight
                    
                    if weight_total > 0:
                        term_score = (weighted_sum / weight_total) * 1000
                
                scores.append(max(1, int(term_score)))  # Ensure minimum 1
                
            except Exception as e:
                continue
        
        # Sort by timestamp and limit to meaningful data
        if timestamps and scores:
            df = pd.DataFrame({'timestamp': timestamps, 'count': scores})
            df = df.sort_values('timestamp')
            
            # Filter out zeros and take last 25 time windows for better visualization
            df = df[df['count'] > 0].tail(25)
            
            if len(df) > 0:
                return df
        
    except Exception as e:
        pass
    
    # Fallback to simulated growth pattern based on trend score
    base_date = pd.Timestamp('2024-01-01')
    timestamps = pd.date_range(base_date, periods=15, freq='W')
    
    # Create realistic growth pattern
    base_score = 10
    growth_factor = 1.15
    scores = [max(1, int(base_score * (growth_factor ** i) + random.randint(-5, 15))) 
              for i in range(len(timestamps))]
    
    return pd.DataFrame({'timestamp': timestamps, 'count': scores})

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

options = filtered["name"].unique()

if len(options) == len(filtered):  # means no filtering yet
    label = "Start typing to search trends..."
else:
    label = "Choose a trend"

trend_choice = st.selectbox(label, options)

# Define the dialog function
@st.dialog("Trend Analysis", width = "large")
def show_trend_details(trend_name):
    st.title(f"📊 {trend_name}")
    
    # Get trend data
    trend_row = df_trends[df_trends['name'] == trend_name].iloc[0] if len(df_trends[df_trends['name'] == trend_name]) > 0 else None
    
    # Create row with time plot and insights
    col1, col2, col3 = st.columns([6, 0.5, 3.5])
    
    with col1:
        # Use real temporal data if available
        if st.session_state.get('data_source') == 'real' and st.session_state.get('results_data'):
            real_ts = create_real_temporal_data(trend_name, st.session_state.results_data)
            
            # Check if we got real temporal data or fallback data
            is_real_data = len(real_ts) > 0 and real_ts['timestamp'].min() < pd.Timestamp('2023-01-01')
            
            if is_real_data:
                fig = px.line(real_ts, x="timestamp", y="count", 
                             title=f"Real Temporal Trend: {trend_name}", 
                             labels={"timestamp": "Date", "count": "TF-IDF Score (×1000)"})
                
                # Add data source info
                date_range = f"{real_ts['timestamp'].min().strftime('%Y-%m-%d')} to {real_ts['timestamp'].max().strftime('%Y-%m-%d')}"
                st.caption(f"📊 Real temporal data from {len(real_ts)} time windows ({date_range})")
            else:
                fig = px.line(real_ts, x="timestamp", y="count", 
                             title=f"Simulated Growth Pattern: {trend_name}", 
                             labels={"timestamp": "Date", "count": "Trend Score"})
                st.caption("📈 Simulated growth pattern (specific term not found in temporal data)")
        else:
            # Fallback to sample data
            fig = px.line(ts, x="timestamp", y="count", title=f"Growth of {trend_name}", labels = {"timestamp": "Date", "count": "Mentions"})
            st.caption("📊 Sample data (real data not available)")
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        st.subheader("🔍 Insights")
        
        if st.session_state.get('data_source') == 'real' and trend_row is not None:
            # Real data insights
            category = trend_row['category']
            tfidf_score = trend_row.get('tfidf_score', 0)
            trend_score = trend_row['trend_score']
            
            st.write(f"**Category:** {category}")
            st.write(f"**TF-IDF Score:** {tfidf_score:.4f}")
            st.write(f"**Trend Score:** {trend_score}")
            st.write(f"- '{trend_name}' shows **strong relevance** in {category.lower()} discussions")
            st.write(f"- Based on **real social media data** analysis")
            st.write(f"- Identified through **advanced NLP modeling**")
            
            # Add temporal growth insights if real data available
            if st.session_state.get('results_data'):
                real_ts = create_real_temporal_data(trend_name, st.session_state.results_data)
                if len(real_ts) >= 2:
                    first_val = real_ts['count'].iloc[0]
                    last_val = real_ts['count'].iloc[-1]
                    if first_val > 0:
                        real_growth = ((last_val - first_val) / first_val) * 100
                        st.write(f"- **Temporal Growth:** {real_growth:.1f}% over time period")
                        st.write(f"- **Peak Score:** {real_ts['count'].max()}")
            
            # Add decay analysis if available
            if st.session_state.get('decay_data'):
                decay_data = st.session_state.decay_data
                categories = decay_data.get('detailed_analysis', {}).get('categories', {})
                
                if category.lower() in categories:
                    cat_data = categories[category.lower()]
                    accelerating_count = cat_data.get('accelerating_terms_count', 0)
                    total_terms = cat_data.get('total_terms_analyzed', 0)
                    
                    if accelerating_count > 0:
                        st.write(f"- **{accelerating_count}/{total_terms}** terms accelerating in {category}")
                    
                    # Check if this specific term is accelerating
                    top_accelerating = cat_data.get('top_accelerating_terms', [])
                    for acc_term in top_accelerating:
                        if acc_term.get('term', '').lower() == trend_name.lower():
                            growth_rate = acc_term.get('avg_growth_rate', 0) * 100
                            st.write(f"- **🚀 ACCELERATING TREND** with {growth_rate:.1f}% growth rate!")
                            break
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