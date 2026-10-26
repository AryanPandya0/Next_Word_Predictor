"""
Script to create the local training corpus 'dataset.txt'.
Combines public-domain text from Project Gutenberg (The Adventures of Sherlock Holmes by Arthur Conan Doyle)
with everyday conversational sentences to provide natural suggestions for both
general conversation and narrative writing.
"""
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def create_dataset():
    print("Fetching Project Gutenberg public-domain text (The Adventures of Sherlock Holmes)...")
    url = "https://www.gutenberg.org/cache/epub/1661/pg1661.txt"
    book_text = ""
    
    try:
        response = requests.get(url, verify=False, timeout=25, headers={'User-Agent': 'Mozilla/5.0'})
        if response.status_code == 200:
            raw_text = response.text
            # Strip Gutenberg license headers and footers
            start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
            end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
            
            start_pos = raw_text.find(start_marker)
            if start_pos != -1:
                start_pos = raw_text.find("\n", start_pos) + 1
            else:
                start_pos = 0
                
            end_pos = raw_text.find(end_marker)
            if end_pos == -1:
                end_pos = len(raw_text)
                
            book_text = raw_text[start_pos:end_pos].strip()
            print(f"Downloaded {len(book_text)} characters from Project Gutenberg.")
        else:
            print(f"Non-200 status code: {response.status_code}")
    except Exception as e:
        print(f"Error downloading Gutenberg text: {e}")

    # Everyday conversational and common English sentences
    # Ensures common everyday prompts like "I am going to/home/there", "How are you", "Thank you very much"
    # have robust representation in the statistical model counts.
    conversational_sentences = """
I am going to the market to buy some groceries for dinner.
I am going to school with my friends every morning.
I am going to work early tomorrow morning.
I am going to college to attend the lecture on natural language processing.
I am going to the library to study for my upcoming exam.
I am going to visit my grandparents this weekend.
I am going to start working on the project right away.
I am going home after finishing all my work at the office.
I am going home to take some rest because I feel tired.
I am going home to spend time with my family and relax.
I am going home now as it is getting dark outside.
I am going home early today because of the weather.
I am going there to meet my classmates for group study.
I am going there right now to see what happened.
I am going there tomorrow morning as planned.
I am going there with my family next week.
I am going there because they invited us for celebration.
I want to thank you for your help and guidance.
Thank you very much for your kind support and assistance.
Thank you very much for everything you have done for us.
Thank you very much indeed for your valuable advice.
Thank you so much for the quick response and feedback.
How are you doing today my dear friend.
How are you feeling this fine morning.
How are you planning to spend your weekend.
What is the best way to solve this difficult problem.
What is the main idea behind statistical language modeling.
What is your name and where do you come from.
It is a wonderful opportunity to learn artificial intelligence.
It is a great day to work on programming and natural language processing.
It is important to understand the concept of bigrams and trigrams.
It was the best of times it was the worst of times.
We need to complete our project before the deadline.
We have to submit the assignment before tomorrow afternoon.
She said that she would be waiting at the train station.
He said that everything was going to be alright.
They are going to participate in the annual college competition.
Can you please help me understand this concept better.
I would like to know more about this interesting topic.
The quick brown fox jumps over the lazy dog.
Natural language processing is an exciting field of computer science.
Statistical language models estimate probabilities of word sequences based on frequency.
N-gram models use the Markov assumption to predict the next word in a sequence.
Bigrams look at the previous one word to predict the next word.
Trigrams look at the previous two words to predict the next word.
I will see you tomorrow at the college campus.
Please let us know if you have any questions or concerns.
Good morning everyone and welcome to our project demonstration.
Have a great day ahead and enjoy your learning journey.
"""

    combined_text = conversational_sentences.strip() + "\n\n" + book_text

    # Clean lines
    cleaned_lines = []
    for line in combined_text.splitlines():
        line = line.strip()
        if line:
            cleaned_lines.append(line)

    final_corpus = "\n".join(cleaned_lines)

    output_path = "dataset.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_corpus)

    print(f"Successfully generated {output_path} with {len(final_corpus)} characters.")

if __name__ == "__main__":
    create_dataset()
