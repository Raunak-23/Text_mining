import streamlit as st
import pandas as pd
import pickle
import os
from streamlit_autorefresh import st_autorefresh

st_autorefresh(interval=900000, key="datarefresh")

st.set_page_config(page_title="GDELT Narrative Tracker", layout="wide")
st.title("Real-Time News Narrative Clusters")

# Path to the data saved by main.py
DATA_PATH = "models/narrative_state.pkl"

if os.path.exists(DATA_PATH):
    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f)
    
    # Convert metadata to a displayable DataFrame
    meta = data['meta']
    df = pd.DataFrame.from_dict(meta, orient='index')
    
    # Sort data once for both the table and the chart
    df = df.sort_values(by='count', ascending=False)

    st.subheader("Top Trending Narratives")
    st.dataframe(df[['label', 'count']], use_container_width=True)

    # Combined & Sorted Bar Chart
    st.subheader("Narrative Volume Distribution")
    st.bar_chart(df.set_index('label')['count'])

else:
    st.warning("Waiting for main.py to generate the first narrative clusters...")