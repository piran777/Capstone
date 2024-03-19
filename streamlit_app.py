import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import test  # Adjust this import according to the actual location of test.py in your project structure

def generate_and_display_heatmap():
    
    #This function is responsible for invoking the heatmap generation functionality
    #from test.py and displaying the resulting heatmap within the Streamlit app.

    # Load the generated HTML file
    with open("traffic_map.html", "r", encoding="utf-8") as f:
        html_content = f.read()

    # Use Streamlit components to render the HTML content
    components.html(html_content, height=600, scrolling=True)

# App main page
st.title('Traffic Heatmap Visualization')

# Button to generate and display the heatmap
if st.button('Generate Heatmap'):
    generate_and_display_heatmap()