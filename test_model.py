from predictor import NGramPredictor

p = NGramPredictor("dataset.txt")
print("Model Stats:", p.get_statistics())

tests = [
    "I am going",
    "going",
    "",
    "xyzunknownword",
    "How are",
    "Thank you",
    "What is the"
]

for t in tests:
    res = p.predict_next_word(t)
    print(f"\n--- Input: '{t}' ---")
    print(f"Model: {res['model_used']}")
    print(f"Context: {res['context']}")
    print(f"Predictions: {[(item['word'], item['percentage'], item['count']) for item in res['predictions']]}")
    print(f"Status: {res['status_message']}")
