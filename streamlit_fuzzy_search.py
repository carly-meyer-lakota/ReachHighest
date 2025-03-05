import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from rapidfuzz import process, fuzz
from difflib import SequenceMatcher

# Download necessary WordNet data
nltk.download('wordnet')
nltk.download('omw-1.4')  # Multilingual support
nltk.download('punkt')  # Tokenization support

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("reach_higher_curriculum_all_units.csv")

data = load_data()

# Define search categories
topic_columns = ["Unit Name", "Vocabulary Words", "Genres"]
concept_columns = ["Language Skill", "Thinking Map Skill", "Reading Skill", "Grammar Skill", "Phonics Skill", "Project"]
all_columns = ["RH Level", "Unit Number", "Unit Name", "Language Skill", "Vocabulary Words", "Thinking Map Skill", "Reading Skill", "Genres", "Grammar Skill", "Project", "Phonics Skill"]

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

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

# Function to lemmatize and tokenize a phrase
def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())  # Tokenize and convert to lowercase
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]  # Lemmatization
    return " ".join(sorted(lemmatized_tokens))  # Sort tokens alphabetically

# Function to get synonyms, hypernyms, and hyponyms
def get_expanded_terms(word):
    terms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            terms.add(lemma.name().replace("_", " "))  # Synonyms
        for hypernym in syn.hypernyms():
            terms.add(hypernym.name().split('.')[0])  # Broader terms
        for hyponym in syn.hyponyms():
            terms.add(hyponym.name().split('.')[0])  # Narrower terms
    return list(terms)

# Function to compute similarity using RapidFuzz and difflib
def compute_similarity(query, text):
    if pd.isna(text):
        return 0
    processed_query = preprocess_text(query)
    processed_text = preprocess_text(str(text))

    # Use RapidFuzz for fuzzy matching
    rf_score = fuzz.token_sort_ratio(processed_query, processed_text)

    # Use difflib's SequenceMatcher for approximate similarity
    diff_score = SequenceMatcher(None, processed_query, processed_text).ratio() * 100  # Convert to percentage

    # Return the maximum of the two scores
    return max(rf_score, diff_score)

# Function to search for best matches
def fuzzy_search(query, data, columns, threshold=50):
    matches = []
    for _, row in data.iterrows():
        best_score = 0
        for col in columns:
            if pd.notna(row[col]):
                score = compute_similarity(query, row[col])
                best_score = max(best_score, score)
        if best_score >= threshold:
            matches.append({'score': best_score, 'row': row})
    return matches

# Expanded search function with synonyms and related terms
def expanded_search(query, data, columns):
    expanded_terms = get_expanded_terms(query) + [query]  # Include original query
    results = []
    for term in expanded_terms:
        results += fuzzy_search(term, data, columns)
    return sorted(results, key=lambda x: x['score'], reverse=True)

# Highlight fuzzy match in results
def highlight_fuzzy_match(text, query):
    if pd.isna(text) or not query:
        return text
    processed_query = preprocess_text(query)
    processed_text = preprocess_text(str(text))

    # Find match using difflib
    match = SequenceMatcher(None, processed_query, processed_text).find_longest_match(0, len(processed_query), 0, len(processed_text))
    
    if match.size > 0:
        matched_text = processed_text[match.b: match.b + match.size]
        pattern = re.compile(re.escape(matched_text), re.IGNORECASE)
        return pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", str(text))
    
    return text

# Perform search on button click
if st.button("Search"):
    matches = []

    if topic_query:
        matches = expanded_search(topic_query, data, topic_columns)

    if concept_query:
        matches += expanded_search(concept_query, data, concept_columns)

    # Sort matches by relevance
    matches.sort(reverse=True, key=lambda x: x['score'])

    if matches:
        st.write(f"Found {len(matches)} matches")
        for match in matches:
            row = match['row']
            st.markdown(f"### Grade Level: {row['RH Level']}, Unit {row['Unit Number']} - {row['Unit Name']}")
            st.markdown("---")
            for col in all_columns:
                if col in topic_columns or col in concept_columns or col in ["RH Level", "Unit Number", "Unit Name"]:
                    query = topic_query if col in topic_columns else concept_query
                    highlighted_text = highlight_fuzzy_match(row[col], query)
                    st.markdown(f"**{col}:** {highlighted_text}", unsafe_allow_html=True)
                else:
                    st.markdown(f"**{col}:** {row[col]}")
            st.markdown("---")
    else:
        st.write("No matches found.")
