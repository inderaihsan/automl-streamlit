import requests
import streamlit as st
from datetime import datetime


from helper import get_local_url, get_server_url
# Streamlit file uploader
local_url = get_local_url()
server_url = get_server_url()




st.write("""
This application allows you to upload geospatial data in Excel format and generate enriched features based on proximity to various points of interest in Jakarta, including:

- Distance to bus stops
- Distance to universities
- Distance to malls
- Distance to toll gates
- Distance to schools
- Distance to roads
- Distance to airports
- Distance to industrial zones
- Proximity to major city in Jawa Tengah (Including Yogyakarta)

Simply upload your file containing longitude and latitude columns, press the 'Process Data' button, and download the processed file with the newly generated features.
""")
if "uploaded_file" not in st.session_state:
    st.session_state["uploaded_file"] = None

# File uploader
uploaded_file = st.file_uploader("Upload your file", type=["xlsx"])

if uploaded_file:
    st.session_state["uploaded_file"] = uploaded_file
    st.write("File uploaded successfully! Please press the 'Process Data' button below to proceed.")

# Check if a file is already uploaded in the session state
if st.session_state["uploaded_file"] is not None:
    # Add a button to process the data
    if st.button("Process Data"):
        # Send file to the Django server
        response = requests.post(
            server_url + '/feat_gen/gen_feat_batam/',
            files={'file': st.session_state["uploaded_file"]},
            verify=False
        )

        # Handle the response
        if response.status_code == 200:
            st.success("Feature Generated!")

            # Generate a timestamp for the file name
            original_file_name = uploaded_file.name.rsplit('.', 1)[0]  # Remove extension
            file_name = f"{original_file_name}_generated_feature_batam.xlsx"

            # Provide download button for the processed file
            st.download_button(
                label="Download Processed Data",
                data=response.content,
                file_name=file_name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.error("Error: Something went wrong! Please ensure the file contains longitude and latitude in the columns.")
