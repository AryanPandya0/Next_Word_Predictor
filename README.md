# Next Word Predictor ✍️
### An NLP-based Statistical Language Model

A simple, beginner-friendly **Next Word Prediction System** built using **Natural Language Processing (NLP)** principles in Python. Similar to smartphone keyboard suggestions, this application analyzes user input in real time and predicts the top three most likely next words using statistical **Bigram** and **Trigram** language models with a back-off strategy.

This project uses **pure statistical NLP** (frequency counting, conditional probabilities, and maximum likelihood estimation) without requiring large language models, external AI APIs, or deep learning frameworks.

---

## 📌 Project Highlights

* **Pure Statistical NLP:** Implements Bigram and Trigram models from scratch using Python standard libraries and NLTK for preprocessing.
* **Real-Time Interactive UI:** Clean, light-colored, student-friendly interface built with [Streamlit](https://streamlit.io/).
* **One-Click Word Insertion:** Clickable suggestion buttons append the predicted word directly to the text input with proper spacing.
* **Mathematical Transparency:** Displays the exact formula, context, frequency count, and conditional probability percentage for each prediction.
* **Offline Operation:** Trains from a local English text corpus (`dataset.txt`). Requires **no internet connection** after initial setup.
* **Viva & Exam Ready:** Includes structured documentation and algorithm breakdowns tailored for undergraduate academic evaluations.

---

## 📁 Project Structure

```text
d:/NLP/
├── app.py              # Streamlit web application & interactive UI
├── predictor.py        # Core NLP logic: tokenization, N-gram counts, probabilities, back-off
├── dataset.txt         # Local English training corpus (~106,000 words)
├── build_dataset.py    # Offline utility script used to assemble the local dataset
├── requirements.txt    # Python package dependencies
└── README.md           # Project documentation, setup guide, and algorithm explanation
```

### Module Responsibilities

| File | Purpose |
| :--- | :--- |
| [`app.py`](file:///d:/NLP/app.py) | Streamlit user interface, input handling, interactive suggestion buttons, live metrics display, and probability tables. |
| [`predictor.py`](file:///d:/NLP/predictor.py) | Class `NGramPredictor` managing text normalization, regex tokenization, frequency counts, probability computation, and Trigram-to-Bigram back-off. |
| [`dataset.txt`](file:///d:/NLP/dataset.txt) | Offline training corpus stored directly in the repository. |
| [`requirements.txt`](file:///d:/NLP/requirements.txt) | List of minimal required libraries (`streamlit`, `nltk`, `requests`). |

---

## ⚙️ Setup and Execution Guide (Windows)

Follow these step-by-step instructions to run the project on a Windows computer:

### Step 1: Open Terminal / PowerShell
Open **PowerShell** or **Command Prompt** and navigate to the project directory:
```powershell
cd d:\NLP
```

### Step 2: Create a Virtual Environment (Recommended)
```powershell
python -m venv venv
```

### Step 3: Activate the Virtual Environment
* On **PowerShell**:
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
  *(If PowerShell displays an execution policy error, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` first)*
* On **Command Prompt (cmd)**:
  ```cmd
  venv\Scripts\activate.bat
  ```

### Step 4: Install Required Packages
```powershell
pip install -r requirements.txt
```

### Step 5: Run the Streamlit Application
```powershell
streamlit run app.py
```

### Step 6: View in Browser
The application will launch automatically in your default web browser. If not, open your browser and navigate to:
```text
http://localhost:8501
```

---

## 🧠 NLP Algorithm & Mathematical Logic

### 1. The Language Modeling Problem
A **Statistical Language Model (SLM)** calculates the joint probability of a sequence of words $W = (w_1, w_2, \dots, w_n)$:

$$P(w_1, w_2, \dots, w_n) = P(w_1) \cdot P(w_2 \mid w_1) \cdot P(w_3 \mid w_1, w_2) \cdots P(w_n \mid w_1, \dots, w_{n-1})$$

Conditioning a word on its entire preceding history becomes computationally expensive and suffers from extreme data sparsity as sentences grow longer.

### 2. The Markov Assumption
To solve this, we apply the **Markov Assumption**: the probability of an upcoming word depends only on the previous $N-1$ words.

* **Bigram Model ($N=2$, First-Order Markov):**
  Looks back **1 word**. The conditional probability of next word $w_i$ given previous word $w_{i-1}$ is:
  $$P(w_i \mid w_{i-1}) = \frac{\text{Count}(w_{i-1}, w_i)}{\sum_{w} \text{Count}(w_{i-1}, w)} = \frac{\text{Count}(w_{i-1}, w_i)}{\text{Count}(w_{i-1})}$$

* **Trigram Model ($N=3$, Second-Order Markov):**
  Looks back **2 words**. The conditional probability of next word $w_i$ given previous two words $(w_{i-2}, w_{i-1})$ is:
  $$P(w_i \mid w_{i-2}, w_{i-1}) = \frac{\text{Count}(w_{i-2}, w_{i-1}, w_i)}{\sum_{w} \text{Count}(w_{i-2}, w_{i-1}, w)} = \frac{\text{Count}(w_{i-2}, w_{i-1}, w_i)}{\text{Count}(w_{i-2}, w_{i-1})}$$

### 3. Back-off Strategy
Natural text contains combinations not seen during training. To handle this safely:
1. **Trigram Step:** When the user types $\ge 2$ words, the model checks if the last two words $(w_{n-1}, w_n)$ exist as a context in the Trigram table.
2. **Bigram Fallback:** If $(w_{n-1}, w_n)$ is not found in the Trigram counts, the model backs off to the Bigram model using only the immediate last word $w_n$.
3. **Single Word:** When the user enters only 1 word, the Bigram model is queried directly.
4. **Out-of-Vocabulary (OOV) / Unseen Context:** If neither model has seen the context, the system displays a clear, helpful message without crashing.

---

## 🛠️ Natural Language Processing Techniques Used

1. **Text Normalization:**
   * All characters are converted to lowercase so words like `"The"` and `"the"` are recognized as the same vocabulary token.
2. **Regex Word Tokenization:**
   * Uses `RegexpTokenizer(r"[a-zA-Z']+")` from NLTK.
   * Preserves contractions such as `don't`, `it's`, and `I'll` while filtering out numbers and punctuation marks.
3. **Sentence Boundary Preservation:**
   * The training text is segmented into sentences using punctuation delimiters (`.`, `!`, `?`).
   * N-grams are generated strictly *within* individual sentences to prevent false cross-sentence transitions (e.g., the last word of sentence 1 predicting the first word of sentence 2).
4. **Frequency Counting & Efficient Indexing:**
   * Unigrams are counted with `collections.Counter`.
   * Bigrams and Trigrams are stored in hash maps (`collections.defaultdict(Counter)`), allowing $O(1)$ context lookups.
5. **Ranking & Filtering:**
   * Candidate words are sorted in descending order of conditional probability $P(w \mid \text{context})$.
   * The top 3 candidates are selected and displayed.

---

## 📚 Dataset Information

* **Source:** A curated offline corpus combining:
  1. Public-domain literary text from **Project Gutenberg** (*The Adventures of Sherlock Holmes* by Sir Arthur Conan Doyle).
  2. Everyday conversational English sentences covering standard greetings, requests, and dialogue.
* **Corpus Metrics:**
  * **Total Tokens:** ~106,365 words
  * **Vocabulary Size:** ~7,860 unique words
  * **Bigram Types:** ~46,332 unique pairs
  * **Trigram Types:** ~77,107 unique triplets
* **Zero Startup Downloads:** The corpus is saved directly as `dataset.txt`. The application loads from this local file and does not require an active internet connection.

---

## 🧪 Example Test Inputs and Expected Outputs

| Input Sentence | Context Used | Active Model | Top 3 Predicted Words & Probabilities |
| :--- | :--- | :--- | :--- |
| **"I am going"** | `('am', 'going')` | **Trigram** | 1. **to** (33.3%)<br>2. **home** (23.8%)<br>3. **there** (23.8%) |
| **"going"** | `'going'` | **Bigram** | 1. **to** (34.7%)<br>2. **on** (12.2%)<br>3. **home** (10.2%) |
| **"How are"** | `('how', 'are')` | **Trigram** | 1. **you** (100.0%) |
| **"Thank you"** | `('thank', 'you')` | **Trigram** | 1. **said** (21.4%)<br>2. **very** (21.4%)<br>3. **for** (14.3%) |
| **"What is the"** | `('is', 'the')` | **Trigram** | 1. **first** (6.6%)<br>2. **most** (3.9%)<br>3. **only** (3.9%) |
| **"xyzunknownword"** | `'xyzunknownword'` | *None* | *No matching N-gram found in training corpus.* |

---

## ⚠️ Limitations & Future Improvements

### Limitations of Basic N-Gram Models
1. **Limited Context Window:** Trigrams can only look back 2 words. They cannot remember themes, subjects, or context from earlier in the paragraph.
2. **Zero-Frequency Problem (Data Sparsity):** If a word combination did not appear in the training text, its probability is calculated as zero, even if it is grammatically valid.
3. **Lack of Semantic Understanding:** N-grams rely entirely on surface word counts without understanding synonyms or sentence meaning.

### Future Improvements
1. **Smoothing Techniques:** Implementing **Laplace (Add-1) Smoothing**, **Good-Turing Estimation**, or **Kneser-Ney Smoothing** to assign small probabilities to unseen N-grams.
2. **Linear Interpolation:** Combining Unigram, Bigram, and Trigram probabilities using learned weights:
   $$P_{\text{interpolated}}(w_3 \mid w_1, w_2) = \lambda_1 P(w_3 \mid w_1, w_2) + \lambda_2 P(w_3 \mid w_2) + \lambda_3 P(w_3)$$
3. **Neural Language Models:** Upgrading from statistical counting to Recurrent Neural Networks (RNNs), LSTMs, or Transformers (e.g. BERT/GPT) for deep contextual understanding.
