"""
Utility functions for Hangman AI
"""
import numpy as np
import re


def load_words(filepath):
    """Load words from a text file."""
    with open(filepath, 'r') as f:
        words = [line.strip().upper() for line in f if line.strip()]
    return words


def get_masked_word(word, guessed_letters):
    """Return word with unguessed letters masked as '_'."""
    return ''.join([letter if letter in guessed_letters else '_' for letter in word])


def encode_state(masked_word, guessed_letters, lives, hmm_probs=None):
    """
    Encode the current game state into a feature vector.
    
    Args:
        masked_word: Current pattern (e.g., '_A__E')
        guessed_letters: Set of guessed letters
        lives: Remaining wrong guesses allowed
        hmm_probs: Probability distribution over letters (optional)
    
    Returns:
        Encoded state vector
    """
    # Basic features
    features = []
    
    # Word length
    features.append(len(masked_word))
    
    # Number of revealed letters
    revealed = sum(1 for c in masked_word if c != '_')
    features.append(revealed)
    
    # Remaining lives
    features.append(lives)
    
    # Number of guessed letters
    features.append(len(guessed_letters))
    
    # If HMM probabilities provided, add top probabilities
    if hmm_probs is not None:
        features.extend(hmm_probs)
    
    return np.array(features)


def letter_to_index(letter):
    """Convert letter (A-Z) to index (0-25)."""
    if not letter.isupper() or not letter.isalpha():
        return -1  # Invalid letter
    return ord(letter) - ord('A')


def index_to_letter(index):
    """Convert index (0-25) to letter (A-Z)."""
    if not (0 <= index < 26):
        return None
    return chr(index + ord('A'))


def get_word_pattern(word):
    """Get position-based pattern for a word."""
    return [(i, c) for i, c in enumerate(word)]


def filter_words_by_pattern(words, pattern, guessed_letters):
    """
    Filter words that match the current pattern and don't contain wrong guesses.
    
    Args:
        words: List of candidate words
        pattern: Current masked pattern (e.g., '_A__E')
        guessed_letters: Set of all guessed letters
    
    Returns:
        List of matching words
    """
    matching = []
    correct_letters = set(c for c in pattern if c != '_')
    wrong_letters = guessed_letters - correct_letters
    
    for word in words:
        if len(word) != len(pattern):
            continue
        
        # Check if word contains wrong letters
        if any(letter in word for letter in wrong_letters):
            continue
        
        # Check if pattern matches
        match = True
        for i, c in enumerate(pattern):
            if c != '_' and word[i] != c:
                match = False
                break
            elif c == '_' and word[i] in correct_letters:
                match = False
                break
        
        if match:
            matching.append(word)
    
    return matching
