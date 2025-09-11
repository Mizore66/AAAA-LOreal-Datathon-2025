import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import random

# Load data here
# -----------------------
# 1. Fake Data for Dry Run
# -----------------------
import random

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
    name, category = random.choice(base_trends)
    rows.append({
        "trend_id": i,
        "name": f"{name}_{i}",  # make names unique
        "trend_score": random.randint(60, 100),
        "platform": random.choice(platforms),
        "category": category,
        "demographic": random.choice(demographics),
        "sentiment": random.choice(sentiments),
        "sentiment_score": round(random.uniform(0.2, 0.95), 2),
        "keywords": f"keywords for {name.lower()}_{i}"
    })

df_trends = pd.DataFrame(rows)

df_trends["platform"] = df_trends["platform"].str.strip().str.title()
df_trends["category"] = df_trends["category"].str.strip().str.title()
df_trends["demographic"] = df_trends["demographic"].str.strip()

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

# applies filters
filtered = df_trends.copy()
if platform:
    filtered = filtered[filtered["platform"].isin(platform)]
if category:
    filtered = filtered[filtered["category"].isin(category)]
if demographic:
    filtered = filtered[filtered["demographic"].isin(demographic)]

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
trend_choice = st.selectbox("Choose a trend", filtered["name"].unique())

# Define the dialog function
@st.dialog("Trend Analysis", width = "large")
def show_trend_details(trend_name):
    st.title(f"📊 {trend_name}")
    
    # Create row with time plot and insights
    col1, col2, col3 = st.columns([6, 0.5, 3.5])
    
    with col1:
        fig = px.line(ts, x="timestamp", y="count", title=f"Growth of {trend_name}", labels = {"timestamp": "Date", "count": "Mentions"})

        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        st.subheader("🔍 Insights")
        st.write(f"- {trend_name} is currently **growing 25% week-over-week**.")
        st.write(f"- Popular among **Millennials** with **80% positive sentiment**.")


# Button to trigger the dialog
if st.button("View Details", disabled=not trend_choice):
    # call function here to retrieve the trend data. smth like retrieve_trend_data(trend_choice)
    # for now the name is provided as input
    # smth like 
    show_trend_details(trend_choice)