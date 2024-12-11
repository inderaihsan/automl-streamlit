import pandas as pd 
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
import io



st.header("""About the App
This app is designed to help users efficiently clean and match data between two datasets, such as asset and tower information from Excel sheets. It automates the process of text standardization, data cleaning, and matching using advanced algorithms, making it ideal for handling large and complex datasets.""")

# Common words to remove
common_words = ['TANAH', 'SUTT', '150', 'KV', 'TANAH;SUTT', '150kV', 'TOWER', "500kV", "Kv", "kV", "150", "500", "Tower-p",
                '150KV', 'TANAH;SUTET', '500KV', 'TANAH;TOWER;SUTT', 'TANAH;TOWER', "SUTET", "Tanah", "Tapak", "Tower"
                'TJ', 'TAPAK', 'TANAH;', 'nan', '500 kV', 'SUTET', 'Sutet', 'sutet', 'SUTT', 'sutt', 'Sutt', 'T.', ';', '#', "+CMGSII", "Tower-"]

# Function to remove common words

def extract_numbers(text):
    if pd.isna(text):
        return ""  # Return an empty string for NaN values
    numbers = re.findall(r'\d+', text) # Find all numbers as strings
    return numbers[-1].lstrip('0') if numbers else ""  # Combine numbers into a single string  # Return at least a single 0 if no numbers

def remove_common_words(text):
    if pd.isna(text):  # Check if the text is NaN
        return text

    # Remove common words
    for word in common_words:
        text = text.replace(word, '')  # Replace the common word with an empty string

    # Remove numbers using a regex
    text = re.sub(r'\d+', '', text)  # Replace digits with an empty string

    return text.strip()  # Remove leading/trailing spaces



def remove_words_with_special_chars(text):
    """
    Removes words containing special characters from the input string, 
    excluding the hyphen (-) as an exception.

    Args:
        text (str): The input string.

    Returns:
        str: The cleaned string.
    """
    # Regular expression to match words with special characters (excluding '-')
    cleaned_text = re.sub(r'\b\S*[!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~]+\S*\b', '', text)
    # Remove extra spaces left by removed words
    return ' '.join(cleaned_text.split())

# Example usage

# Apply the function to the 'Nama Aset 1' column

def remove_special_characters(text):
    """
    Removes all special characters from the input string, excluding spaces.

    Args:
        text (str): The input string.

    Returns:
        str: The string with special characters removed.
    """
    # Regular expression to remove all special characters except spaces
    return re.sub(r'[^\w\s-]', '', text)

def remove_floating_words(text):
    """
    Removes isolated single-character and two-character words from a string.

    Parameters:
        text (str): The input string.

    Returns:
        str: The cleaned string with floating words removed.
    """
    # Use regex to find and remove isolated one- and two-character words
    cleaned_text = re.sub(r'\b\w{1,2}\b', '', text)

    # Clean up any extra spaces that may result
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    return cleaned_text




from difflib import SequenceMatcher

def jaccard_similarity_char_level(str1, str2, n=1):
    """Calculates the Jaccard similarity between two strings at the character level (or n-gram level)."""
    # Create sets of characters (or n-grams) from both strings
    set1 = set([str1[i:i+n] for i in range(len(str1)-n+1)])
    set2 = set([str2[i:i+n] for i in range(len(str2)-n+1)])

    # Calculate the intersection and union
    intersection = set1.intersection(set2)
    union = set1.union(set2)

    # Return the Jaccard similarity (intersection / union)
    if not union:
        return 0
    return len(intersection) / len(union)

def edit_similarity(str1, str2):
    """Calculates the edit similarity (normalized Levenshtein distance)."""
    return SequenceMatcher(None, str1, str2).ratio()

def match_combined_columns(df1, df2, n_range=range(1, 2)):
    """
    Matches rows between two DataFrames based on the 'combined' column using a combination of
    Jaccard similarity and edit similarity over a range of n-gram values to find the maximum similarity.

    Args:
        df1 (pd.DataFrame): First DataFrame with a 'combined' column.
        df2 (pd.DataFrame): Second DataFrame with a 'combined' column.
        n_range (range): Range of n values for n-gram Jaccard similarity.

    Returns:
        pd.DataFrame: Updated df1 with additional columns for the best match, similarity score, and n value.
    """
    # Lists to store the results
    best_matches = []
    similarity_scores = []
    best_n_values = []

    # Iterate through each 'combined' in df1
    i = 0
    loading_bar = st.progress(value = 0, text = "Doing string matching algorithm...")
    for combined1 in df1['combined']:
        i = i+1
        loading_bar.progress(i/len(df1), text='matching in progress')
        # Initialize tracking for best match
        best_match = None
        best_similarity = -1
        best_n = None

        # Compare with each 'combined' in df2
        for combined2 in df2['combined']:
            for n in n_range:
                # Calculate Jaccard similarity
                jaccard_sim = jaccard_similarity_char_level(combined1, combined2, n=n)

                # Calculate edit similarity
                edit_sim = edit_similarity(combined1, combined2)

                # Combine similarities (weighted average can be adjusted as needed)
                combined_similarity = 0.5 * jaccard_sim + 0.5 * edit_sim

                # Update best match if similarity is higher
                if combined_similarity > best_similarity:
                    best_similarity = combined_similarity
                    best_match = combined2
                    best_n = n

        # Append the best match, similarity score, and n value
        best_matches.append(best_match)
        similarity_scores.append(best_similarity)
        best_n_values.append(best_n)

    # Add results to df1
    df1['best_match'] = best_matches
    df1['similarity_score'] = similarity_scores
    df1['best_n'] = best_n_values

    # Merge additional information from df2 based on the best match
    matched_df = df1.merge(
        df2,
        left_on='best_match',
        right_on='combined',
        how='left',
        suffixes=('_df1', '_df2')
    )

    return matched_df
@st.cache_data
def load_pln_data(data) : 
    objek_penilaian = pd.read_excel(data, sheet_name = 'ObjekPenilaianTT')
    matchtofa = pd.read_excel(data, sheet_name = 'MatchToFA')
    objek_penilaian['clean_name'] = objek_penilaian['Nama Aset 1'].apply(remove_common_words)
    objek_penilaian['clean_name'] = objek_penilaian['clean_name'].apply(remove_words_with_special_chars)
    objek_penilaian['clean_name'] = objek_penilaian['clean_name'].apply(remove_special_characters)
    objek_penilaian['clean_name'] = objek_penilaian['clean_name'].apply(remove_floating_words)
    objek_penilaian['number_extracted'] = objek_penilaian['Nama Aset 1'].apply(extract_numbers)
    objek_penilaian['combined'] = objek_penilaian['clean_name'] +' '+ objek_penilaian['number_extracted']
    matchtofa['clean_name'] = matchtofa['Nama Tower'].apply(remove_common_words)
    matchtofa['clean_name'] = matchtofa['clean_name'].apply(remove_words_with_special_chars)
    matchtofa['clean_name'] = matchtofa['clean_name'].apply(remove_special_characters)
    matchtofa['clean_name'] = matchtofa['clean_name'].apply(remove_floating_words)
    matchtofa['number_extracted'] = matchtofa['Nama Tower'].apply(extract_numbers)
    matchtofa['combined'] = matchtofa['clean_name'] +' ' + matchtofa['number_extracted'] 
    return objek_penilaian , matchtofa

uploaded = st.file_uploader("Please upload your Excel file", type=['xlsx'])  
if uploaded:
    if "processed_data" not in st.session_state:
        # Load and process data if not already cached in session state
        objek_penilaian, matchtofa = load_pln_data(uploaded)
        b__ = match_combined_columns(objek_penilaian, matchtofa)
        b__['Nama Saluran_df1'] = b__['Nama Saluran_df2']
        b__['Y_df1'] = b__['Y_df2']
        b__['X_df1'] = b__['X_df2']
        b__['JarakTower (m)_df1'] = b__['JarakTower (m)_df2']
        for col in b__.columns:
            if '_df1' in col:
                b__.rename(columns={col: col.replace("_df1", " ")}, inplace=True)
            if '_df2' in col:
                b__.drop(col, axis=1, inplace=True)
        st.session_state["processed_data"] = b__
    else:
        # Retrieve cached data from session state
        b__ = st.session_state["processed_data"]

    if not b__.empty:
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            b__.to_excel(writer, index=False, sheet_name='Sheet1')
            writer.close()
            buffer.seek(0)

        # Provide a download button
        st.download_button(
            label="Download Calculation result here!",
            data=buffer,
            file_name="regression_result_with_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    



