import requests
import streamlit as st
from datetime import datetime


from helper import get_local_url, get_server_url
# Streamlit file uploader
local_url = get_local_url()
server_url = get_server_url()




st.write("""
This application allows you to upload geospatial data in Excel format and generate prediction using Machine Learning Model of Bali Region

Simply upload your file (Make sure all feature exist in the file), press the 'Process Data' button, and download the processed file with Prediction.

the process might take up several minutes depending on the complexity of the model and the number of data
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
            server_url + '/feat_gen/predict/bali/',
            files={'file': st.session_state["uploaded_file"]},
            verify=False
        )

        # Handle the response
        if response.status_code == 200:
            st.success("Sucessfuly predict the file! check (prediction column)!")

            # Generate a timestamp for the file name
            original_file_name = uploaded_file.name.rsplit('.', 1)[0]  # Remove extension
            file_name = f"{original_file_name}_predicted.xlsx"

            # Provide download button for the processed file
            st.download_button(
                label="Download Processed Data",
                data=response.content,
                file_name=file_name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            try : 
                error_message = response.json().get('Message', 'An error occurred')
                feat__ = response.json().get('feat__', 'An error occurred')
                st.error(error_message)
                st.error(feat__)
            except : 
                st.error("Snap!, file is too large to handle!")
            # st.error(response.data['feat__'])
