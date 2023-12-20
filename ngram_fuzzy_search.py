import pickle
from collections import defaultdict
from nltk import ngrams
from fuzzywuzzy import process, fuzz

# File path to store and load the trigram index
INDEX_FILE_PATH = "data/trigram_index.pkl"
TARGET_WORDS_FILE_PATH = "data/dataset.pkl"

# Load the target words from the file
try:
    with open(TARGET_WORDS_FILE_PATH, "rb") as file:
        target_words = pickle.load(file)
except FileNotFoundError:
    target_words = []

# Load the trigram index from the file if it exists
try:
    with open(INDEX_FILE_PATH, "rb") as file:
        trigram_index = pickle.load(file)
except FileNotFoundError:
    trigram_index = defaultdict(list)

def save_trigram_index():
    # Save the trigram index to the file
    with open(INDEX_FILE_PATH, "wb") as file:
        pickle.dump(trigram_index, file)

# Populate the initial trigram index if it's empty
if not trigram_index:
    for word in target_words:
        trigrams = [''.join(gram) for gram in ngrams(word, 2)]
        for trigram in trigrams:
            trigram_index[trigram].append(word)
    save_trigram_index()

def save_target_words():
    # Save the target words to the file
    with open(TARGET_WORDS_FILE_PATH, "wb") as file:
        pickle.dump(target_words, file)

def phonetic_match(word1, word2, threshold=60):
    # Use the ratio method from fuzz to calculate similarity
    return fuzz.ratio(word1, word2) > threshold

def fuzzy_search(search_term, threshold=60):
    search_term = ''.join(filter(str.isalpha, search_term)).lower()
    trigrams = [''.join(gram) for gram in ngrams(search_term, 2)]

    # Get potential candidates from the trigram index
    candidates = set()
    for trigram in trigrams:
        candidates.update(trigram_index.get(trigram, []))

    # Perform a fuzzy search on potential candidates
    results = process.extract(search_term, candidates, limit=5)

    # Filter the results based on improved phonetic matching
    phonetic_results = [(word, score) for word, score in results if phonetic_match(search_term, word, threshold)]
    print(phonetic_results)
    return phonetic_results

# insert new words from array of words
def insert(word):
    word = ''.join(filter(str.isalpha, word)).lower()
    trigrams = [''.join(gram) for gram in ngrams(word, 2)]
    for trigram in trigrams:
        trigram_index[trigram].append(word)

    # Save the updated trigram index to the file
    save_trigram_index()

# replace word from old to new
def update(old_word, new_word):
    old_word = ''.join(filter(str.isalpha, old_word)).lower()
    new_word = ''.join(filter(str.isalpha, new_word)).lower()
    # Remove old spelling from the trigram index
    trigrams_old = [''.join(gram) for gram in ngrams(old_word, 2)]
    for trigram in trigrams_old:
        if old_word in trigram_index[trigram]:
            trigram_index[trigram].remove(old_word)

    # Add new spelling to the trigram index
    trigrams_new = [''.join(gram) for gram in ngrams(new_word, 2)]
    for trigram in trigrams_new:
        trigram_index[trigram].append(new_word)

    # Save the updated trigram index to the file
    save_trigram_index()

# delete a word
def delete(word):
    word = ''.join(filter(str.isalpha, word)).lower()
    # Remove old spelling from the trigram index
    trigrams_old = [''.join(gram) for gram in ngrams(word, 2)]
    for trigram in trigrams_old:
        if word in trigram_index[trigram]:
            trigram_index[trigram].remove(word)

    save_trigram_index()
