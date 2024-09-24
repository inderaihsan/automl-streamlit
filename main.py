import streamlit as st

# Set the title of the app
st.title("Welcome to the Data Science Hub!")

# Add a welcoming subtitle
st.subheader("A platform for Regression Analysis, ML Model Training, and Prediction")

# Add a greeting message
st.write("""
Welcome! 👋 This application allows you to:
- Perform **regression analysis** to explore relationships between variables.
- Train **machine learning models** using your data.
- Make predictions using the trained models.

Use the sidebar to navigate between different sections.
""")

# Add a friendly button to guide users to start
if st.button("Get Started"):
    st.write("Head over to the sidebar to explore the available tools!")

# Provide some footer information or a welcome note
st.info("Feel free to upload your data and models, and let the magic of data science happen!")
