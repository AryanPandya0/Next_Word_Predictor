"""
predictor.py
============
Core Natural Language Processing (NLP) module for Next Word Prediction.
Implements a statistical N-gram language model (Unigram, Bigram, and Trigram)
using conditional frequency counts and maximum likelihood estimation (MLE).

Designed mini-projects: clean, fully commented, and viva-ready.
"""

import re
from collections import defaultdict, Counter
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.util import bigrams, trigrams


class NGramPredictor:
    """
    Statistical N-gram Language Model for Next Word Prediction.
    
    Mathematical Principles:
    ------------------------
    1. Bigram Model (Markov Assumption - 1st order):
       P(w_i | w_{i-1}) = Count(w_{i-1}, w_i) / Count(w_{i-1})
       
    2. Trigram Model (Markov Assumption - 2nd order):
       P(w_i | w_{i-2}, w_{i-1}) = Count(w_{i-2}, w_{i-1}, w_i) / Count(w_{i-2}, w_{i-1})
       
    3. Back-off Strategy:
       - If the last 2 words match a Trigram context, predict using the Trigram model.
       - If no Trigram match is found (or only 1 word is available), fall back to Bigram.
       - If no Bigram match exists, safely report that no prediction is available.
    """

    def __init__(self, corpus_path="dataset.txt"):
        """
        Initializes the predictor, tokenizes the corpus, and builds N-gram models.
        
        Args:
            corpus_path (str): Path to the local training text file.
        """
        self.corpus_path = corpus_path
        
        # Word tokenizer that keeps alphanumeric words and apostrophes (e.g., "don't", "it's")
        self.tokenizer = RegexpTokenizer(r"[a-zA-Z']+")
        
        # Frequency count storage:
        # unigram_counts: word -> total frequency count
        self.unigram_counts = Counter()
        
        # bigram_counts: w1 -> Counter({w2: count})
        self.bigram_counts = defaultdict(Counter)
        
        # trigram_counts: (w1, w2) -> Counter({w3: count})
        self.trigram_counts = defaultdict(Counter)
        
        # Total counts per context for fast probability calculation:
        # bigram_context_totals: w1 -> sum(counts of all w2 following w1)
        self.bigram_context_totals = defaultdict(int)
        
        # trigram_context_totals: (w1, w2) -> sum(counts of all w3 following (w1, w2))
        self.trigram_context_totals = defaultdict(int)
        
        # Corpus statistics for UI and viva reporting
        self.total_tokens = 0
        self.vocab_size = 0
        self.total_bigram_types = 0
        self.total_trigram_types = 0
        
        # Load and train
        self.train_from_corpus(corpus_path)

    def preprocess_text(self, text):
        """
        Cleans and tokenizes text into lowercase word tokens.
        
        Args:
            text (str): Raw input text.
            
        Returns:
            list[str]: Cleaned list of lowercase tokens.
        """
        if not text:
            return []
        tokens = self.tokenizer.tokenize(text.lower())
        return tokens

    def train_from_corpus(self, corpus_path):
        """
        Reads the local corpus, splits into sentences, and constructs N-gram counts.
        Sentence-level splitting ensures N-grams do not improperly bridge
        across unrelated sentences.
        
        Args:
            corpus_path (str): Filepath of the training corpus.
        """
        try:
            with open(corpus_path, "r", encoding="utf-8") as f:
                corpus_text = f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Training corpus not found at '{corpus_path}'. Please check file path.")

        # Split corpus into sentences using standard punctuation markers (. ! ?)
        sentences = re.split(r"[.!?]+", corpus_text)

        all_tokens = []

        for sentence in sentences:
            tokens = self.preprocess_text(sentence)
            if not tokens:
                continue

            all_tokens.extend(tokens)

            # 1. Unigram counts
            self.unigram_counts.update(tokens)

            # 2. Bigram counts using nltk.util.bigrams
            for w1, w2 in bigrams(tokens):
                self.bigram_counts[w1][w2] += 1
                self.bigram_context_totals[w1] += 1

            # 3. Trigram counts using nltk.util.trigrams
            for w1, w2, w3 in trigrams(tokens):
                context = (w1, w2)
                self.trigram_counts[context][w3] += 1
                self.trigram_context_totals[context] += 1

        # Calculate dataset summary metrics
        self.total_tokens = len(all_tokens)
        self.vocab_size = len(self.unigram_counts)
        self.total_bigram_types = sum(len(followers) for followers in self.bigram_counts.values())
        self.total_trigram_types = sum(len(followers) for followers in self.trigram_counts.values())

    def predict_next_word(self, user_input, top_k=3):
        """
        Predicts the top-k most probable next words given user input.
        
        Decision Pipeline:
        1. Tokenize and lowercase user input.
        2. If 2 or more words:
           - Check Trigram model using the last two words (w_{n-1}, w_n).
           - If matches found, calculate conditional probabilities and return top-k.
           - If no match, fall back to Bigram using the last word (w_n).
        3. If 1 word:
           - Use Bigram model on the word (w_n).
        4. If 0 words:
           - Return empty result with prompt message.
        5. If neither model finds candidates:
           - Return empty result with helpful explanation.
           
        Args:
            user_input (str): The current text typed by the user.
            top_k (int): Number of top predictions to return (default 3).
            
        Returns:
            dict: {
                "predictions": list of dicts [{"word": str, "probability": float, "percentage": str, "count": int}],
                "model_used": "Trigram" | "Bigram" | None,
                "context": tuple or str,
                "status_message": str
            }
        """
        tokens = self.preprocess_text(user_input)

        if not tokens:
            return {
                "predictions": [],
                "model_used": None,
                "context": None,
                "status_message": "Type a word or sentence above to see statistical next-word suggestions."
            }

        # Step 1: Try Trigram Model if at least 2 words exist
        if len(tokens) >= 2:
            w1, w2 = tokens[-2], tokens[-1]
            context = (w1, w2)

            if context in self.trigram_counts and self.trigram_counts[context]:
                candidates = self.trigram_counts[context]
                total = self.trigram_context_totals[context]

                # Calculate conditional probability: P(w3 | w1, w2) = Count(w1, w2, w3) / Count(w1, w2)
                ranked = sorted(
                    candidates.items(),
                    key=lambda item: (-item[1], item[0])
                )[:top_k]

                predictions = [
                    {
                        "word": word,
                        "probability": round(count / total, 4),
                        "percentage": f"{(count / total) * 100:.1f}%",
                        "count": count
                    }
                    for word, count in ranked
                ]

                return {
                    "predictions": predictions,
                    "model_used": "Trigram",
                    "context": f"('{w1}', '{w2}')",
                    "status_message": f"Predictions generated using Trigram Model based on context: '{w1} {w2}'."
                }

        # Step 2: Fallback to Bigram Model (or primary for 1-word input)
        last_word = tokens[-1]
        if last_word in self.bigram_counts and self.bigram_counts[last_word]:
            candidates = self.bigram_counts[last_word]
            total = self.bigram_context_totals[last_word]

            # Calculate conditional probability: P(w2 | w1) = Count(w1, w2) / Count(w1)
            ranked = sorted(
                candidates.items(),
                key=lambda item: (-item[1], item[0])
            )[:top_k]

            predictions = [
                {
                    "word": word,
                    "probability": round(count / total, 4),
                    "percentage": f"{(count / total) * 100:.1f}%",
                    "count": count
                }
                for word, count in ranked
            ]

            note = "Trigram had no matching history; backed off to Bigram." if len(tokens) >= 2 else "Single-word context; using Bigram Model."

            return {
                "predictions": predictions,
                "model_used": "Bigram",
                "context": f"'{last_word}'",
                "status_message": f"{note} Based on previous word: '{last_word}'."
            }

        # Step 3: No matching N-gram found in training corpus
        return {
            "predictions": [],
            "model_used": None,
            "context": f"'{last_word}'",
            "status_message": f"No matching N-gram found in the training corpus for '{last_word}'. Try another common English word."
        }

    def get_statistics(self):
        """
        Returns corpus and model summary metrics for display in the UI.
        """
        return {
            "total_tokens": self.total_tokens,
            "vocab_size": self.vocab_size,
            "bigram_types": self.total_bigram_types,
            "trigram_types": self.total_trigram_types
        }
