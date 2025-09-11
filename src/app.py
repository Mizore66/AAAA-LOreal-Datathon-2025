import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

# Load data here
# -----------------------
# 1. Fake Data for Dry Run
# -----------------------
df_trends = pd.DataFrame({
    "trend_id": list(range(1, 11)),
    "name": [
        "#GlowSkin", "#SunProtection", "Viral Audio 123", "#MakeupTutorial",
        "#HairCareHack", "Viral Dance Beat", "#EcoBeauty", "ASMR Lipstick Test",
        "#NoFilterChallenge", "Viral Audio XYZ"
    ],
    "trend_score": [92, 85, 78, 88, 74, 83, 80, 76, 90, 82],
    "platform": [
        "TikTok", "Instagram", "TikTok", "YouTube",
        "Instagram", "TikTok", "Twitter", "YouTube",
        "TikTok", "Instagram"
    ],
    "category": [
        "Skincare", "Makeup", "Lifestyle", "Makeup",
        "Hair", "Audio", "Skincare", "Makeup",
        "Lifestyle", "Audio"
    ],
    "demographic": [
        "Gen Z", "Millennials", "Gen Z", "Millennials",
        "Gen Z", "Gen Z", "Millennials", "Gen Z",
        "Gen Z", "Millennials"
    ],
    "sentiment": [
        "positive", "neutral", "negative", "positive",
        "positive", "neutral", "positive", "negative",
        "positive", "neutral"
    ],
    "sentiment_score": [0.8, 0.5, 0.2, 0.9, 0.75, 0.55, 0.85, 0.3, 0.95, 0.45],
    "keywords": [
        "glow skin serum", "spf sunscreen uv", "funny meme remix", "beginner eye shadow",
        "silky hair mask", "dance challenge audio", "sustainable skincare", "lipstick swatch review",
        "raw selfie trend", "remix viral beat"
    ]
})

ts = pd.DataFrame({
    "timestamp": pd.date_range("2025-09-01", periods=10, freq="D"),
    "count": [5, 10, 15, 40, 60, 90, 120, 180, 150, 130]
})

# View
st.set_page_config(layout="wide")
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
            padding-left: 10rem;
            padding-right: 10rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("TrendSpotter Dashboard")

leaderboard, gap, filters = st.columns([6.5, 0.25, 2.75]) #row

with filters:
    st.subheader("Filters")
    platform = st.multiselect("Platform", df_trends["platform"].unique())
    category = st.multiselect("Category", df_trends["category"].unique())
    demographic = st.multiselect("Demographic", df_trends["demographic"].unique())

# Apply filters
filtered = df_trends.copy()
if platform:
    filtered = filtered[filtered["platform"].isin(platform)]
if category:
    filtered = filtered[filtered["category"].isin(category)]
if demographic:
    filtered = filtered[filtered["demographic"].isin(demographic)]

with leaderboard:
    # Trends leaderboard
    st.subheader("Top Emerging Trends")
    st.dataframe(
        filtered[["name", "platform", "category", "trend_score"]].sort_values("trend_score", ascending=False).rename(
            columns = {
                "name": "Trend Name / Hashtag",
                "platform": "Platform",
                "category": "Category",
                "trend_score": "Trend Score"
            }
        ))

# Main search/selection interface
# Initialize session state for the selected trend
if 'selected_trend' not in st.session_state:
    st.session_state.selected_trend = None

# Main search interface
st.subheader("Trend Search")
trend_choice = st.selectbox("Choose a trend", filtered["name"].unique())

# Define the dialog function
@st.dialog("Trend Analysis", width = "large")
def show_trend_details(trend_name):
    st.title(f"📊 {trend_name}")
    
    # Create row with time plot and insights
    col1, col2, col3 = st.columns([6, 0.5, 3.5])
    
    with col1:
        fig = px.line(ts, x="timestamp", y="Number of interactions", title=f"Growth of {trend_name}")
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        st.subheader("🔍 Insights")
        st.write(f"- {trend_name} is currently **growing 25% week-over-week**.")
        st.write(f"- Popular among **Millennials** with **80% positive sentiment**.")


# Button to trigger the dialog
if st.button("View Details", disabled=not trend_choice):
    show_trend_details(trend_choice)