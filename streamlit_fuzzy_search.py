import streamlit as st
import pandas as pd
from fuzzywuzzy import process
import re
import nltk
from nltk.corpus import wordnet

# Download necessary WordNet data
nltk.download('wordnet')
nltk.download('omw-1.4')  # For multilingual WordNet support, if needed

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

# Functions for synonym and related term expansion
def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)

def get_related_terms(word):
    related_terms = set()
    for syn in wordnet.synsets(word):
        # Add hypernyms (broader terms)
        for hypernym in syn.hypernyms():
            related_terms.add(hypernym.name().split('.')[0])
        # Add hyponyms (narrower terms)
        for hyponym in syn.hyponyms():
            related_terms.add(hyponym.name().split('.')[0])
    return list(related_terms)

# Expanded search function
def expanded_search(query, data):
    expanded_terms = get_synonyms(query) + get_related_terms(query)  # Combine synonyms and related terms
    results = []
    for term in expanded_terms:
        # Perform fuzzy search for each expanded term
        results += fuzzy_search(term, data)
    return sorted(results, key=lambda x: x['relevance'], reverse=True)

# Fuzzy search function
def fuzzy_search(query, data):
    matches = []
    for _, row in data.iterrows():
        best_score = 0
        for col in all_columns:
            if pd.notna(row[col]):
                score = process.extractOne(query, [str(row[col])])[1]
                best_score = max(best_score, score)
        if best_score > 50:
            matches.append({'score': best_score, 'row': row})
    return matches

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
        matches = expanded_search(topic_query, data)  # Use expanded search for topics
   
    if concept_query:
        matches = expanded_search(concept_query, data)  # Use expanded search for concepts
   
    # Sort matches by relevance
    matches.sort(reverse=True, key=lambda x: x['score'])
   
    st.write(f"Found {len(matches)} matches")
   
    for match in matches:
        row = match['row']
        st.markdown(f"### Grade Level: {row['RH Level']}, Unit {row['Unit Number']} - {row['Unit Name']}")
        st.markdown("---")
        for col in all_columns:
            if col in topic_columns or col in concept_columns or col in ["RH Level", "Unit Number", "Unit Name"]:
                highlighted_text = highlight_fuzzy_match(row[col], topic_query if col in topic_columns else concept_query)
                st.markdown(f"**{col}:** {highlighted_text}", unsafe_allow_html=True)
            else:
                st.markdown(f"**{col}:** {row[col]}")
        st.markdown("---")
