import streamlit as st
import pandas as pd
from fuzzywuzzy import process
import re

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("reach_higher_curriculum_all_units.csv")

data = load_data()

# Define search categories
topic_columns = ["Unit Name", "Vocabulary Words", "Genres"]
concept_columns = ["Language Skill", "Thinking Map Skill", "Reading Skill", "Grammar Skill", "Phonics Skill", "Project"]
all_columns = ["RH Level", "Unit Number", "Unit Name", "Language Skill", "Vocabulary Words", "Thinking Map Skill", "Reading Skill", "Genres", "Grammar Skill", "Project", "Phonics Skill"]

# Streamlit UI
st.title("Reach Higher Curriculum Search")

# Two search boxes side by side
col1, col2 = st.columns(2)

if 'topic_query' not in st.session_state:
    st.session_state['topic_query'] = ""
if 'concept_query' not in st.session_state:
    st.session_state['concept_query'] = ""

def clear_other(search_type):
    if search_type == 'topic':
        st.session_state['concept_query'] = ""
    else:
        st.session_state['topic_query'] = ""

topic_query = col1.text_input("Search by Topic", value=st.session_state['topic_query'], on_change=clear_other, args=("topic",))
concept_query = col2.text_input("Search by Concept", value=st.session_state['concept_query'], on_change=clear_other, args=("concept",))

def highlight_fuzzy_match(text, query):
    if pd.isna(text) or not query:
        return text
    matches = process.extract(query, [text], limit=1)
    if matches and matches[0][1] > 50:  # If match score is above threshold
        match_text = matches[0][0]
        pattern = re.compile(re.escape(match_text), re.IGNORECASE)
        return pattern.sub(lambda match: f"<mark>{match.group(0)}</mark>", str(text))
    return text

if st.button("Search"):
    matches = []
   
    if topic_query:
        for _, row in data.iterrows():
            best_score = 0
            for col in topic_columns:
                if pd.notna(row[col]):
                    score = process.extractOne(topic_query, [str(row[col])])[1]
                    best_score = max(best_score, score)
            if best_score > 50:
                matches.append((best_score, row, 'topic'))
   
    if concept_query:
        for _, row in data.iterrows():
            best_score = 0
            for col in concept_columns:
                if pd.notna(row[col]):
                    score = process.extractOne(concept_query, [str(row[col])])[1]
                    best_score = max(best_score, score)
            if best_score > 50:
                matches.append((best_score, row, 'concept'))
   
    # Sort matches by relevance
    matches.sort(reverse=True, key=lambda x: x[0])
   
    st.write(f"Found {len(matches)} matches")
   
    for score, row, match_type in matches:
        with st.container():
            st.markdown(f"### Grade Level: {row['RH Level']}, Unit {row['Unit Number']} - {row['Unit Name']}")
            st.markdown("---")
            for col in all_columns:
                if (match_type == 'topic' and col in topic_columns) or (match_type == 'concept' and col in concept_columns) or col in ["RH Level", "Unit Number", "Unit Name"]:
                    highlighted_text = highlight_fuzzy_match(row[col], topic_query if match_type == 'topic' else concept_query)
                    st.markdown(f"**{col}:** {highlighted_text}", unsafe_allow_html=True)
                else:
                    st.markdown(f"**{col}:** {row[col]}")
            st.markdown("---")
