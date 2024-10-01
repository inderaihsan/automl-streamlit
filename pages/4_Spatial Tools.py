import streamlit as st
import joblib
import pandas as pd
import numpy as np 
import geopandas as gpd 
import folium
import mapclassify
import matplotlib.pyplot as plt
from streamlit_folium import st_folium
from helper import create_coordinate_2_dots, transform_data_to_geodataframe , remove_inf
import io 

@st.cache_data
def load_data(data) : 
    return pd.read_excel(data)

def transform_to_dataframe(name, longitude, latitude) : 
    longitude = float(longitude)
    latitude = float(latitude)
    data = pd.DataFrame({
        'Name' : name, 
        'longitude' : longitude, 
        'latitude' : latitude
    }, index = [0]) 
    return data
st.header("An interface for Geopandas for performing distance calculation")
st.subheader("Introduction")
st.markdown("""Welcome to the Geopandas Distance Calculation Interface, a user-friendly application designed to facilitate spatial analysis by calculating distances between geographical points. In this app, users can easily upload two datasets containing the latitude and longitude of locations. The application will visualize these points on an interactive map, showcasing their geographical distribution. Once the files are uploaded, the app computes the distance between the specified locations, providing valuable insights for geographical studies, urban planning, and logistics. Additionally, users can download an Excel template to help format their data for seamless integration with the tool. Experience the power of spatial analysis at your fingertips!""") 
# example_data = pd.DataFrame(
#     {
#     'Nama' : 'Indonesia Stock Exchange' ,
#     'latitude' : -6.2235445768191, 
#     'longitude' : 106.80866846085952
#     }, index = [0]
# )

# example_data2 = pd.DataFrame(
#     {
#     'Nama' : 'Monumen Nasional' ,
#     'latitude' : -6.175328344464615, 
#     'longitude' : 106.82709457250273
#     }, index = [0]
# )

# col1, col2 = st.columns(2)
# with col1 : 
#     example_data2_geo = transform_data_to_geodataframe(example_data2) 
#     st_folium(example_data2_geo.explore(style_kwds = dict(radius = 10)), width = 450, height = 450)
#     # st.text("map of The national monument of Indonesia")
# with col2 : 
#     example_data_geo = transform_data_to_geodataframe(example_data) 
#     st_folium(example_data_geo.explore(style_kwds = dict(radius = 10)), width = 450, height = 450)
#     # st.text("map of Indonesian stock exchange")
# output = io.BytesIO()
# with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
#         example_data.to_excel(writer, index = False)
# st.download_button(
#     label="Download excel template for this tools",
#     data=output.getvalue(),
#     file_name="template_for_distance_tools.xlsx",
#     mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
# )

    
# Create two columns side by side
col4, col5 = st.columns(2)

# Column 4 (From Location)
with col4:
    st.subheader("Calculate distance from: ")
    
    # Checkbox for using a point instead of uploading a file
    point_from = st.checkbox("I want to use point instead", key='point_from')
    
    if not point_from:
        # File uploader for the "From" file
        df_from = st.file_uploader("Upload your file", type=["xlsx"], key='df_from')
        
        if df_from:
            try:
                df_from = load_data(df_from)
                df_from['longitude'] = pd.to_numeric(df_from['longitude'], errors='coerce')
                df_from['latitude'] = pd.to_numeric(df_from['latitude'], errors='coerce')
                df_from = transform_data_to_geodataframe(df_from)
                show_map_from = st.checkbox("View map preview:", key='show_map_from')
                if show_map_from:
                    preview_map_from = df_from.sample(frac=0.1, random_state=42)
                    with st.form(key="smth"):
                        st_folium(preview_map_from.explore(), height=450, width=450)
                        st.form_submit_button("Map preview")
            except:
                st.error("There's no longitude or latitude column in the data")
    else:
        # Text inputs for manually entering a point (coordinates and name)
        coordinate_name_from = st.text_input("Name: ", key='coordinate_name_from')
        long_from = st.text_input("Longitude: ", key='long_from')
        lat_from = st.text_input("Latitude: ", key='lat_from')
        
        if long_from and lat_from and coordinate_name_from:
            try:
                # Convert the entered point into a GeoDataFrame
                df_from = transform_to_dataframe(coordinate_name_from, long_from, lat_from)
                st.dataframe(df_from)

                # Checkbox to preview the map
                show_map_from = st.checkbox("View map preview:", key='show_map_from_checkbox')
                if show_map_from:
                    st_folium(transform_data_to_geodataframe(df_from).explore(), height=450, width=450)
            except:
                st.error("Please provide a correct geometry value")

# Column 5 (To Location)
with col5:
    st.subheader("To: ")
    
    # Checkbox for using a point instead of uploading a file
    point_to = st.checkbox("I want to use point instead", key='point_to')
    
    if not point_to:
        # File uploader for the "To" file
        df_to = st.file_uploader("Upload your file", type=["geojson"], key='df_to')
       
        
        if df_to:
            # try:
                # Load and transform the data into a GeoDataFrame
                df_to = gpd.read_file(df_to)
                df_to.to_crs(epsg = 32749, inplace = True)
                show_map_to = st.checkbox("View map preview:", key='show_map_to')
                if show_map_to:
                    with st.form(key="smth2"):
                        st_folium(df_to.explore(), height=450, width=450)
                        st.form_submit_button("Map preview")
            # except:
                # st.error("There's no longitude or latitude column in the data")
    else:
        # Text inputs for manually entering a point (coordinates and name)
        coordinate_name_to = st.text_input("Name: ", key='coordinate_name_to')
        long_to = st.text_input("Longitude: ", key='long_to')
        lat_to = st.text_input("Latitude: ", key='lat_to')
        
        if long_to and lat_to and coordinate_name_to:
            try:
                # Convert the entered point into a GeoDataFrame
                df_to = transform_to_dataframe(coordinate_name_to, long_to, lat_to)
                st.dataframe(df_to)
                df_to = transform_data_to_geodataframe(df_to)
                # Checkbox to preview the map
                show_map_to = st.checkbox("View map preview:", key='show_map_to_checkbox')
                if show_map_to:
                    with st.form(key="smth2"):
                        st_folium(df_to.explore(), height=450, width=450)
                        st.form_submit_button("Map preview")
            except:
                st.error("Please provide a correct geometry value")

def calculate_distance(df1, df2, distance_col) : 
    if("index_left" in df1) : 
        df1.drop("index_left", axis = 1, inplace = True)
    if("index_right" in df1) : 
        df1.drop("index_right", axis = 1, inplace = True)
    if("index_right" in df2) : 
        df2.drop("index_right", axis = 1, inplace = True)  
    if("index_left" in df2) : 
        df2.drop("index_left", axis = 1, inplace = True)
    return gpd.sjoin_nearest(df1,df2, how = 'left', distance_col = distance_col)

if(isinstance(df_from, gpd.GeoDataFrame) and isinstance(df_to, gpd.GeoDataFrame)) : 
    st.subheader("Print ready to calculate")
    column_name = st.text_input("what will be the name of the column?")
    button_click = st.button("Calculate distance from left data to right data", )  
    if(button_click and column_name) : 
        calculated_object = calculate_distance(df_from, df_to, column_name) 
        calculated_object = remove_inf(calculated_object)
        st_folium(calculated_object.explore(column = column_name))
        st.dataframe(calculated_object.drop("geometry", axis = 1))
        
    
  