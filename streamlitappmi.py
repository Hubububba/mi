import streamlit as st
import pandas as pd
import plotly.express as px
import io
import requests
import json # Import json for API payload and parsing

# --- Configuration ---
st.set_page_config(
    page_title="Interactive Media Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a modern, futuristic look (similar to the HTML app)
st.markdown("""
<style>
    .reportview-container {
        background: #0F172A; /* bg-gray-900 */
    }
    .main .block-container {
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #A78BFA; /* purple-400 */
        font-family: 'Inter', sans-serif;
    }
    p, li {
        color: #D1D5DB; /* gray-300 */
        font-family: 'Inter', sans-serif;
    }
    .stFileUploader {
        background-color: #1F2937; /* Darker gray */
        border-radius: 0.75rem;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease-in-out;
    }
    .stFileUploader:hover {
        box-shadow: 0 0 15px rgba(129, 140, 248, 0.6); /* Tailwind indigo-300 with blur */
        transform: translateY(-2px);
    }
    .stButton>button {
        background-color: #8B5CF6; /* purple-600 */
        color: white;
        font-weight: bold;
        padding: 0.75rem 1.5rem;
        border-radius: 0.5rem;
        transition: all 0.3s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #7C3AED; /* purple-700 */
        box-shadow: 0 0 10px rgba(129, 140, 248, 0.4);
    }
    .chart-container {
        background-color: #1F2937; /* Darker gray */
        padding: 1.5rem;
        border-radius: 0.75rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease-in-out;
        margin-bottom: 2.5rem; /* Equivalent to mb-10 */
    }
    .chart-container:hover {
        box-shadow: 0 0 15px rgba(129, 140, 248, 0.6);
        transform: translateY(-2px);
    }
    .stPlotlyChart {
        background-color: #1F2937; /* Same as chart container */
        border-radius: 0.5rem;
    }
    .stExpander {
        border-radius: 0.75rem;
        border: 1px solid #374151; /* gray-700 */
        background-color: #1F2937;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .stExpander > div > div > p {
        font-size: 1rem;
        line-height: 1.5;
        color: #E0E0E0;
        white-space: pre-wrap; /* Preserve formatting for insights */
    }
    .stSpinner > div > span {
        color: #A78BFA;
    }
</style>
""", unsafe_allow_html=True)

# --- API Key for Gemini (Placeholder for Canvas Environment) ---
# In a real Streamlit deployment, you'd use st.secrets.
# For this Canvas environment, the API key will be automatically provided by the backend for the fetch call.
GEMINI_API_KEY = "" # Leave as empty string for Canvas auto-injection

# --- Helper Function for Gemini API Call ---
def call_gemini_api(prompt):
    """Makes a request to the Gemini API to generate insights."""
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
        headers = {'Content-Type': 'application/json'}
        payload = {
            "contents": [
                {"role": "user", "parts": [{"text": prompt}]}
            ]
        }
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status() # Raise an exception for HTTP errors
        result = response.json()

        if result.get('candidates') and result['candidates'][0].get('content') and result['candidates'][0]['content'].get('parts'):
            return result['candidates'][0]['content']['parts'][0]['text']
        else:
            st.error(f"Unexpected API response structure: {result}")
            return "Could not generate insights due to an unexpected API response."
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to generate insights: {e}")
        return "Failed to generate insights due to a network or API error."

# --- 1. Data Cleaning Function ---
def clean_data(df):
    """
    Cleans and normalizes the input DataFrame.
    - Converts 'Date' to datetime.
    - Fills missing 'Engagements' with 0.
    - Normalizes column names.
    """
    # Normalize column names first
    df.columns = [col.lower().replace(' ', '_').strip() for col in df.columns]

    # Check for required columns
    required_columns = ['date', 'platform', 'sentiment', 'location', 'engagements', 'media_type']
    if not all(col in df.columns for col in required_columns):
        st.error(f"Missing one or more required columns. Please ensure your CSV has: {', '.join(required_columns)}")
        return None

    # Convert 'date' to datetime, handling errors
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['date'], inplace=True) # Drop rows where date conversion failed

    # Fill missing 'engagements' with 0 and ensure integer type
    df['engagements'] = pd.to_numeric(df['engagements'], errors='coerce').fillna(0).astype(int)

    # Normalize 'sentiment' to lowercase
    df['sentiment'] = df['sentiment'].astype(str).str.lower()

    # Sort by date for time series charts
    df = df.sort_values('date').reset_index(drop=True)

    return df

# --- 2. Chart Generation Functions ---

def create_sentiment_chart(df):
    """Creates a Plotly pie chart for Sentiment Breakdown."""
    sentiment_counts = df['sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    fig = px.pie(
        sentiment_counts,
        names='Sentiment',
        values='Count',
        title='Sentiment Breakdown',
        hole=0.4,
        color_discrete_map={
            'positive': '#2ECC71', # Green
            'negative': '#E74C3C', # Red
            'neutral': '#F1C40F',  # Yellow
            'unknown': '#7F8C8D'   # Grey
        }
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#E0E0E0',
        title_font_color='#A78BFA',
        legend_font_color='#E0E0E0',
        hoverlabel_font_color='#E0E0E0'
    )
    fig.update_traces(textinfo="percent+label", hoverinfo="label+percent+value", showlegend=True)
    return fig

def create_engagement_trend_chart(df):
    """Creates a Plotly line chart for Engagement Trend over time."""
    # Group by date and sum engagements
    engagement_by_date = df.groupby(df['date'].dt.date)['engagements'].sum().reset_index()
    engagement_by_date.columns = ['Date', 'Total Engagements']
    fig = px.line(
        engagement_by_date,
        x='Date',
        y='Total Engagements',
        title='Engagement Trend Over Time',
        markers=True,
        line_shape='spline', # Smooth line
        color_discrete_sequence=['#3498DB'] # Blue
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#E0E0E0',
        title_font_color='#A78BFA',
        xaxis_title_font_color='#E0E0E0',
        yaxis_title_font_color='#E0E0E0',
        xaxis=dict(gridcolor='#444'),
        yaxis=dict(gridcolor='#444')
    )
    return fig

def create_platform_engagements_chart(df):
    """Creates a Plotly bar chart for Platform Engagements."""
    platform_engagements = df.groupby('platform')['engagements'].sum().sort_values(ascending=False).reset_index()
    platform_engagements.columns = ['Platform', 'Total Engagements']
    fig = px.bar(
        platform_engagements,
        x='Platform',
        y='Total Engagements',
        title='Platform Engagements',
        color='Total Engagements', # Color bars based on engagement
        color_continuous_scale=px.colors.sequential.Plasma # Purple color scale
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#E0E0E0',
        title_font_color='#A78BFA',
        xaxis_title_font_color='#E0E0E0',
        yaxis_title_font_color='#E0E0E0',
        xaxis=dict(gridcolor='#444'),
        yaxis=dict(gridcolor='#444')
    )
    return fig

def create_media_type_mix_chart(df):
    """Creates a Plotly pie chart for Media Type Mix."""
    media_type_counts = df['media_type'].value_counts().reset_index()
    media_type_counts.columns = ['Media Type', 'Count']
    fig = px.pie(
        media_type_counts,
        names='Media Type',
        values='Count',
        title='Media Type Mix',
        hole=0.4,
        color_discrete_map={
            'image': '#1ABC9C', # Teal
            'video': '#F39C12', # Orange
            'text': '#95A5A6',  # Grey
            'other': '#D35400'  # Dark Orange
        }
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#E0E0E0',
        title_font_color='#A78BFA',
        legend_font_color='#E0E0E0'
    )
    fig.update_traces(textinfo="percent+label", hoverinfo="label+percent+value", showlegend=True)
    return fig

def create_top_locations_chart(df):
    """Creates a Plotly bar chart for Top 5 Locations by Engagements."""
    location_engagements = df.groupby('location')['engagements'].sum().sort_values(ascending=False).head(5).reset_index()
    location_engagements.columns = ['Location', 'Total Engagements']
    fig = px.bar(
        location_engagements,
        x='Location',
        y='Total Engagements',
        title='Top 5 Locations by Engagements',
        color='Total Engagements',
        color_continuous_scale=px.colors.sequential.Greens # Green color scale
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#E0E0E0',
        title_font_color='#A78BFA',
        xaxis_title_font_color='#E0E0E0',
        yaxis_title_font_color='#E0E0E0',
        xaxis=dict(gridcolor='#444'),
        yaxis=dict(gridcolor='#444')
    )
    return fig

# --- Main Streamlit App Logic ---
st.title("Interactive Media Intelligence Dashboard")

st.markdown("---")

st.header("1. Upload Your CSV File")
uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type="csv",
    help="Expected columns: Date, Platform, Sentiment, Location, Engagements, Media Type"
)

if uploaded_file is not None:
    st.success("File uploaded successfully! Processing data...")
    try:
        # Read the CSV data
        data_io = io.BytesIO(uploaded_file.getvalue())
        raw_df = pd.read_csv(data_io)

        # Clean the data
        cleaned_df = clean_data(raw_df.copy())

        if cleaned_df is not None:
            st.markdown("---")
            st.header("2. Data Cleaning & Normalization")
            st.markdown("""
                Your data has been successfully cleaned:
                * 'Date' column converted to datetime objects.
                * Missing 'Engagements' values filled with 0.
                * Column names normalized (e.g., 'Media Type' to 'media_type').
            """)

            st.markdown("---")
            st.header("3. Interactive Charts & 4. Top Insights")

            # --- Chart 1: Sentiment Breakdown ---
            st.subheader("Sentiment Breakdown")
            with st.container():
                st.plotly_chart(create_sentiment_chart(cleaned_df), use_container_width=True)
                sentiment_counts = cleaned_df['sentiment'].value_counts().to_dict()
                sentiment_prompt = f"Based on the following sentiment counts from media data: {sentiment_counts}. Provide top 3 concise insights."
                with st.spinner("Generating insights for Sentiment Breakdown..."):
                    sentiment_insights = call_gemini_api(sentiment_prompt)
                st.markdown(f"**Top 3 Insights:**\n{sentiment_insights}")

            st.markdown("---")

            # --- Chart 2: Engagement Trend over time ---
            st.subheader("Engagement Trend Over Time")
            with st.container():
                st.plotly_chart(create_engagement_trend_chart(cleaned_df), use_container_width=True)
                # Prepare data for prompt: start, end date engagements
                engagement_by_date = cleaned_df.groupby(cleaned_df['date'].dt.date)['engagements'].sum().reset_index()
                if not engagement_by_date.empty:
                    first_date = engagement_by_date.iloc[0]['date'].strftime('%Y-%m-%d')
                    last_date = engagement_by_date.iloc[-1]['date'].strftime('%Y-%m-%d')
                    initial_engagements = engagement_by_date.iloc[0]['Total Engagements']
                    final_engagements = engagement_by_date.iloc[-1]['Total Engagements']
                    trend_prompt = f"Based on engagement data from {first_date} to {last_date}, with initial engagements of {initial_engagements} and final engagements of {final_engagements}. Provide top 3 concise insights about the engagement trend."
                else:
                    trend_prompt = "No engagement data available. Provide top 3 general insights about engagement trends in media analysis."

                with st.spinner("Generating insights for Engagement Trend..."):
                    engagement_insights = call_gemini_api(trend_prompt)
                st.markdown(f"**Top 3 Insights:**\n{engagement_insights}")

            st.markdown("---")

            # --- Chart 3: Platform Engagements ---
            st.subheader("Platform Engagements")
            with st.container():
                st.plotly_chart(create_platform_engagements_chart(cleaned_df), use_container_width=True)
                platform_engagements = cleaned_df.groupby('platform')['engagements'].sum().sort_values(ascending=False).head(5).to_dict()
                platform_prompt = f"Based on platform engagements: {platform_engagements}. Provide top 3 concise insights."
                with st.spinner("Generating insights for Platform Engagements..."):
                    platform_insights = call_gemini_api(platform_prompt)
                st.markdown(f"**Top 3 Insights:**\n{platform_insights}")

            st.markdown("---")

            # --- Chart 4: Media Type Mix ---
            st.subheader("Media Type Mix")
            with st.container():
                st.plotly_chart(create_media_type_mix_chart(cleaned_df), use_container_width=True)
                media_type_counts = cleaned_df['media_type'].value_counts().to_dict()
                media_type_prompt = f"Based on media type counts: {media_type_counts}. Provide top 3 concise insights."
                with st.spinner("Generating insights for Media Type Mix..."):
                    media_type_insights = call_gemini_api(media_type_prompt)
                st.markdown(f"**Top 3 Insights:**\n{media_type_insights}")

            st.markdown("---")

            # --- Chart 5: Top 5 Locations ---
            st.subheader("Top 5 Locations by Engagements")
            with st.container():
                st.plotly_chart(create_top_locations_chart(cleaned_df), use_container_width=True)
                top_locations = cleaned_df.groupby('location')['engagements'].sum().sort_values(ascending=False).head(5).to_dict()
                location_prompt = f"Based on top 5 locations by engagement: {top_locations}. Provide top 3 concise insights."
                with st.spinner("Generating insights for Top 5 Locations..."):
                    location_insights = call_gemini_api(location_prompt)
                st.markdown(f"**Top 3 Insights:**\n{location_insights}")

        else:
            st.error("Data cleaning failed. Please check your CSV file format and column names.")

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
        st.info("Please ensure your CSV file is correctly formatted and contains the expected columns: 'Date', 'Platform', 'Sentiment', 'Location', 'Engagements', 'Media Type'.")
