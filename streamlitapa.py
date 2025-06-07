import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide")
st.title("📊 Interactive Media Intelligence Dashboard – Spryzz")

# Upload CSV
uploaded_file = st.file_uploader("Upload Dataset CSV Anda", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])

    st.subheader("📂 Tinjauan Data")
    st.dataframe(df.head())

    # Sentiment Breakdown
    st.subheader("💬 Sentiment Breakdown")
    sentiment_counts = df['Sentiment'].value_counts().reset_index()
    fig_sentiment = px.pie(sentiment_counts, names='index', values='Sentiment',
                           title='Distribusi Sentimen', hole=0.4)
    st.plotly_chart(fig_sentiment, use_container_width=True)

    # Engagement Trend Over Time
    st.subheader("📈 Engagement Trend Over Time")
    engagement_trend = df.groupby('Date')['Engagements'].sum().reset_index()
    fig_engagement = px.line(engagement_trend, x='Date', y='Engagements',
                             title='Trend Engagement Harian')
    st.plotly_chart(fig_engagement, use_container_width=True)

    # Platform Engagement
    st.subheader("📊 Engagement per Platform")
    platform_engagement = df.groupby('Platform')['Engagements'].sum().reset_index()
    fig_platform = px.bar(platform_engagement, x='Platform', y='Engagements',
                          title='Perbandingan Engagement antar Platform')
    st.plotly_chart(fig_platform, use_container_width=True)

    # Media Type Mix
    st.subheader("🧱 Media Type Mix")
    media_type_mix = df['Media_Type'].value_counts().reset_index()
    fig_media = px.pie(media_type_mix, names='index', values='Media_Type',
                       title='Proporsi Tipe Media')
    st.plotly_chart(fig_media, use_container_width=True)

    # Top 5 Locations by Engagement
    st.subheader("🌍 Top 5 Lokasi dengan Engagement Tertinggi")
    top_locations = df.groupby('Location')['Engagements'].sum().nlargest(5).reset_index()
    fig_location = px.bar(top_locations, x='Engagements', y='Location', orientation='h',
                          title='Lokasi dengan Engagement Tertinggi')
    st.plotly_chart(fig_location, use_container_width=True)
