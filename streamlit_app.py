import streamlit as st
from streamlit_folium import folium_static
import folium
from test import get_decision_for_row
import pandas as pd

# Load in the traffic light locations
traffic_lights_df = pd.read_csv('TrafficLightsLocations_decimal.csv')

# Streamlit layout
st.title("Traffic Intersection Analysis")

# Dictionary of intersection names and corresponding row indices
intersection_dict = traffic_lights_df['Location'].reset_index().set_index('Location')['index'].to_dict()

# Columns to create layout
col1, col2 = st.columns([1, 3]) 

with col1:
    st.subheader("Select an Intersection:")
    # Use a select box for choosing intersections
    option = st.selectbox("Choose an Intersection:", options=traffic_lights_df['Location'].tolist())
    # Get the row index based on the selected intersection name
    row_index = intersection_dict[option]
    if st.button("Analyze Intersection"):
        # Use the row index in the decision line
        decision = get_decision_for_row(row_index)
        st.success(f"Decision: {decision}")  # Displaying the decision

with col2:
    st.subheader("Map View")
    # Use iframe to display hosted heat map
    st.markdown('<iframe src="https://nandrusiw.github.io/Capstone/traffic_map.html" width="640" height="480"></iframe>', unsafe_allow_html=True)

# Custom CSS to attempt to make the sidebar more compact and the map area larger
st.markdown("""
    <style>
    .css-18e3th9 {
        flex: 1;
    }
    .css-1d391kg {
        padding-top: 3rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    </style>
""", unsafe_allow_html=True)