import streamlit as st
import pandas as pd
import zipfile
from io import BytesIO

def split_dataframe(df, parts):
    """
    Splits a DataFrame into approximately equal parts.
    """
    chunk_size = len(df) // parts
    splits = []
    for i in range(parts):
        start = i * chunk_size
        end = start + chunk_size if i < parts - 1 else len(df)
        splits.append(df.iloc[start:end])
    return splits

# Streamlit App
st.title("Split Excel File and Download as ZIP")

uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx", "xls"])

if uploaded_file is not None:
    try:
        # Read the Excel file
        df = pd.read_excel(uploaded_file)
        st.write("Uploaded DataFrame:")
        st.dataframe(df.head())

        # Number of parts to split into
        parts = st.number_input("Number of parts to split into:", min_value=1, max_value=10, value=3, step=1)

        if st.button("Split and Download"):
            split_dfs = split_dataframe(df, parts)

            # Create a ZIP file
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                for i, part in enumerate(split_dfs):
                    # Save each part as an Excel file in the ZIP
                    part_name = f"part_{i + 1}.xlsx"
                    excel_buffer = BytesIO()
                    part.to_excel(excel_buffer, index=False, sheet_name=f"Part {i + 1}")
                    zf.writestr(part_name, excel_buffer.getvalue())

            # Set the buffer position to the beginning
            zip_buffer.seek(0)

            # Provide a download button
            st.download_button(
                label="Download ZIP file",
                data=zip_buffer,
                file_name="split_files_excel.zip",
                mime="application/zip"
            )
    except Exception as e:
        st.error(f"An error occurred: {e}")
