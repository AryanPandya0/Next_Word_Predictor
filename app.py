"""
app.py
======
Streamlit User Interface for Next Word Prediction System.
Demonstrates a statistical N-gram (Bigram & Trigram) language model in NLP.
"""

import streamlit as st
from predictor import NGramPredictor

# Set page configuration - clean, light theme
st.set_page_config(
    page_title="Next Word Predictor | NLP Mini-Project",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom light-theme CSS styling for a clean, academic, student-friendly look
st.markdown("""
<style>
    /* Main container spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 950px;
    }
    
    /* Header typography */
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    
    /* Model badges */
    .badge-trigram {
        background-color: #dbeafe;
        color: #1d4ed8;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-bigram {
        background-color: #fef3c7;
        color: #b45309;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-none {
        background-color: #f1f5f9;
        color: #64748b;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    
    /* Button styling for suggestions */
    .stButton > button {
        border-radius: 6px;
        font-weight: 500;
        border: 1px solid #cbd5e1;
        background-color: #ffffff;
        color: #0f172a;
        transition: all 0.15s ease-in-out;
    }
    .stButton > button:hover {
        border-color: #3b82f6;
        color: #1d4ed8;
        background-color: #f0f7ff;
    }
</style>
""", unsafe_allow_html=True)


# Cache predictor instance so the corpus is loaded only once on startup
@st.cache_resource
def load_predictor():
    return NGramPredictor(corpus_path="dataset.txt")


predictor = load_predictor()
stats = predictor.get_statistics()

# Initialize session state for user text
if "user_text" not in st.session_state:
    st.session_state["user_text"] = "I am going"


def handle_suggestion_click(word):
    """
    Callback when user clicks a suggestion button.
    Appends the clicked word with proper spacing so the user can continue typing.
    """
    current_text = st.session_state["user_text"].rstrip()
    if current_text:
        st.session_state["user_text"] = f"{current_text} {word} "
    else:
        st.session_state["user_text"] = f"{word} "


def handle_clear():
    """Callback to clear the text area."""
    st.session_state["user_text"] = ""


def set_example_sentence(sentence):
    """Callback to set an example sentence."""
    st.session_state["user_text"] = sentence


# ---------------------------------------------------------
# Sidebar: Dataset Information & Corpus Statistics
# ---------------------------------------------------------
with st.sidebar:
    st.header("📊 Corpus Statistics")
    st.caption("Locally trained statistical language model")

    st.metric(label="Total Words (Tokens)", value=f"{stats['total_tokens']:,}")
    st.metric(label="Unique Vocabulary", value=f"{stats['vocab_size']:,} words")
    st.metric(label="Unique Bigrams", value=f"{stats['bigram_types']:,}")
    st.metric(label="Unique Trigrams", value=f"{stats['trigram_types']:,}")

    st.markdown("---")
    st.subheader("💡 Try Example Inputs")
    examples = [
        "I am going",
        "How are",
        "Thank you",
        "What is the",
        "It was the",
        "She said that"
    ]
    for ex in examples:
        st.button(
            f'"{ex}"',
            key=f"ex_{ex}",
            on_click=set_example_sentence,
            args=(ex,)
        )

    st.markdown("---")
    st.caption("College Mini-Project | Natural Language Processing")


# ---------------------------------------------------------
# Main Page: Header
# ---------------------------------------------------------
st.markdown('<div class="main-title">Next Word Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">An NLP-based Statistical Language Model</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# Section 1: Text Input Area
# ---------------------------------------------------------
st.subheader("1. Enter Your Text")

# Directly bound to st.session_state["user_text"]
st.text_area(
    label="Type a sentence below to see real-time word suggestions:",
    key="user_text",
    height=100,
    placeholder="Type something here (e.g. 'I am going', 'How are', 'Thank you')...",
    help="Type naturally or edit the text. Word predictions refresh automatically."
)

col_clear, _ = st.columns([1, 4])
with col_clear:
    st.button("🗑️ Clear Text", on_click=handle_clear)

# ---------------------------------------------------------
# Section 2: Compute Predictions
# ---------------------------------------------------------
current_text = st.session_state.get("user_text", "")
prediction_result = predictor.predict_next_word(current_text, top_k=3)
predictions = prediction_result["predictions"]
model_used = prediction_result["model_used"]
status_msg = prediction_result["status_message"]
context = prediction_result["context"]

# ---------------------------------------------------------
# Section 3: Interactive Suggestions (Clickable Buttons)
# ---------------------------------------------------------
st.subheader("2. Word Suggestions (Top 3)")
st.caption("Click any suggestion to insert it into your sentence:")

if predictions:
    cols = st.columns(3)
    for i, pred in enumerate(predictions):
        with cols[i]:
            button_label = f"➕ {pred['word']} ({pred['percentage']})"
            st.button(
                button_label,
                key=f"pred_btn_{i}_{pred['word']}",
                on_click=handle_suggestion_click,
                args=(pred["word"],)
            )
else:
    st.info(f"ℹ️ {status_msg}")

# ---------------------------------------------------------
# Section 4: Model Decision & Probability Breakdown
# ---------------------------------------------------------
st.markdown("---")
st.subheader("3. Model Decision & Probability Breakdown")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("**Active Prediction Context:**")
    if model_used == "Trigram":
        st.markdown(f'<span class="badge-trigram">Trigram Model Active</span> Context: <code>{context}</code>', unsafe_allow_html=True)
        st.latex(r"P(w_3 \mid w_1, w_2) = \frac{\text{Count}(w_1, w_2, w_3)}{\text{Count}(w_1, w_2)}")
    elif model_used == "Bigram":
        st.markdown(f'<span class="badge-bigram">Bigram Model Active</span> Context: <code>{context}</code>', unsafe_allow_html=True)
        st.latex(r"P(w_2 \mid w_1) = \frac{\text{Count}(w_1, w_2)}{\text{Count}(w_1)}")
    else:
        st.markdown('<span class="badge-none">No Active Model</span>', unsafe_allow_html=True)
        st.write("Type words in the box above to generate predictions.")

    st.caption(status_msg)

with col2:
    if predictions:
        st.markdown("**Candidate Ranking & Probabilities:**")
        table_data = [
            {
                "Rank": f"#{idx + 1}",
                "Predicted Word": p["word"],
                "Frequency": p["count"],
                "Probability (P)": f"{p['probability']:.4f}",
                "Likelihood": p["percentage"]
            }
            for idx, p in enumerate(predictions)
        ]
        st.dataframe(table_data, hide_index=True)
    else:
        st.caption("No probability distribution to display currently.")

# ---------------------------------------------------------
# Section 5: How It Works & NLP Concepts (Viva & Academic Reference)
# ---------------------------------------------------------
st.markdown("---")
with st.expander("📖 How the Prediction Works (Algorithm & Mathematical Logic)", expanded=False):
    st.markdown("""
### 1. Statistical Language Modeling & The Markov Assumption
A **Language Model (LM)** estimates the probability distribution of words in natural text. By the chain rule of probability:
$$P(w_1, w_2, \\dots, w_n) = \\prod_{i=1}^{n} P(w_i \\mid w_1, \\dots, w_{i-1})$$

In practice, conditioning on the full sentence history is computationally intractable. We apply the **Markov Assumption**, which approximates history by looking back only $N-1$ words:
* **Bigram Model ($N=2$, First-order Markov):**
  The probability of the current word depends solely on the single preceding word:
  $$P(w_i \\mid w_{i-1}) = \\frac{\\text{Count}(w_{i-1}, w_i)}{\\text{Count}(w_{i-1})}$$

* **Trigram Model ($N=3$, Second-order Markov):**
  The probability of the current word depends on the previous two words:
  $$P(w_i \\mid w_{i-2}, w_{i-1}) = \\frac{\\text{Count}(w_{i-2}, w_{i-1}, w_i)}{\\text{Count}(w_{i-2}, w_{i-1})}$$

### 2. Back-off Strategy (Handling Unseen Sequences)
1. **Trigram Lookup:** When at least 2 words are entered, the system queries the $(w_{n-1}, w_n)$ context in the Trigram index.
2. **Bigram Fallback:** If the trigram context does not exist in the training data, the model backs off to the Bigram model using only the immediate last word $w_n$.
3. **Graceful Handling:** If neither model contains the history, the system displays a clear message rather than crashing.
    """)

with st.expander("🛠️ Natural Language Processing (NLP) Techniques Used", expanded=False):
    st.markdown("""
* **Corpus Acquisition:** Uses a clean local public-domain text (*The Adventures of Sherlock Holmes* from Project Gutenberg combined with everyday conversational phrases) completely offline.
* **Text Preprocessing & Normalization:** Converts all text to lowercase to ensure uniform vocabulary representations.
* **Tokenization:** Regex-based word tokenization (`[a-zA-Z']+`) captures standard words and contractions (e.g. *don't*, *it's*) while discarding numbers and punctuation.
* **Sentence Boundary Splitting:** Segments the corpus using delimiters (`.`, `!`, `?`) so that transitions between unrelated sentences are not incorrectly learned as N-grams.
* **N-gram Generation:** Builds Bigram $(w_i, w_{i+1})$ and Trigram $(w_i, w_{i+1}, w_{i+2})$ tuples using `nltk.util`.
* **Frequency Counting & Maximum Likelihood Estimation (MLE):** Uses Python dictionaries and `collections.Counter` to calculate conditional probabilities.
* **Probability Ranking:** Sorts candidate words in descending order of probability to present the top three suggestions.
    """)
