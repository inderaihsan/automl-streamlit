import requests
import streamlit as st


# Streamlit file uploader
local_url = "http://127.0.0.1:8000" 
server_url = "https://apiavm.rhr.co.id" 


uploaded_file = st.file_uploader("Upload your file", type=["xlsx"])

if uploaded_file:
    # Send file to the Django server
    response = requests.post(
        server_url+'/feat_gen/gen_feat_jakarta/',
        files={'file': uploaded_file}, 
        verify=False
    )

    # Handle the response
    if response.status_code == 200:
        # Provide download button for the processed file
        st.success("Feature Generated!")
        st.download_button(
            label="Download Processed Data",
            data=response.content,
            file_name="processed_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.error("Error: 'Something went wrong! , please ensure the file contains longitude and latitude in the column'")