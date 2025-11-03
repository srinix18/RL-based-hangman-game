"""
Hidden Markov Model for letter probability prediction
"""
import numpy as np
from collections import defaultdict, Counter
from utils import load_words, filter_words_by_pattern, letter_to_index, index_to_letter


class HangmanHMM:
    """
    HMM-based model for predicting letter probabilities in Hangman.
    Uses frequency-based approach with position-aware statistics.
    """
    
    def __init__(self):
        self.words = []
        self.letter_freq = defaultdict(int)
        self.position_freq = defaultdict(lambda: defaultdict(int))
        self.length_freq = defaultdict(list)
        
    def train(self, corpus_file):
        """Train HMM on corpus of words."""
        print("Training HMM model...")
        self.words = load_words(corpus_file)
        
        # Filter to only keep words with valid letters (A-Z)
        self.words = [word for word in self.words if word.isalpha() and word.isupper()]
        
        # Calculate overall letter frequencies
        for word in self.words:
            for letter in word:
                if letter.isupper() and letter.isalpha():
                    self.letter_freq[letter] += 1
        
        # Calculate position-specific frequencies
        for word in self.words:
            for pos, letter in enumerate(word):
                if letter.isupper() and letter.isalpha():
                    self.position_freq[len(word)][(pos, letter)] += 1
        
        # Group words by length
        for word in self.words:
            self.length_freq[len(word)].append(word)
        
        print(f"HMM trained on {len(self.words)} words")
    
    def predict_letter_probabilities(self, masked_word, guessed_letters):
        """
        Predict probability distribution over unguessed letters.
        
        Args:
            masked_word: Current pattern (e.g., '_A__E')
            guessed_letters: Set of already guessed letters
        
        Returns:
            numpy array of size 26 with probabilities for each letter
        """
        word_length = len(masked_word)
        
        # Filter candidate words matching the pattern
        candidates = filter_words_by_pattern(
            self.length_freq.get(word_length, self.words),
            masked_word,
            guessed_letters
        )
        
        # If no candidates, fall back to general frequency
        if not candidates:
            candidates = self.words
        
        # Count letter frequencies in candidate words
        letter_counts = Counter()
        for word in candidates:
            for letter in set(word):  # Count each letter once per word
                if letter not in guessed_letters and letter.isupper() and letter.isalpha():
                    letter_counts[letter] += 1
        
        # Convert to probability distribution
        probs = np.zeros(26)
        total = sum(letter_counts.values())
        
        if total > 0:
            for letter, count in letter_counts.items():
                if letter.isupper() and letter.isalpha():
                    idx = letter_to_index(letter)
                    if 0 <= idx < 26:  # Safety check
                        probs[idx] = count / total
        else:
            # Uniform distribution if no information
            unguessed_count = sum(1 for i in range(26) 
                                if index_to_letter(i) not in guessed_letters)
            if unguessed_count > 0:
                for i in range(26):
                    if index_to_letter(i) not in guessed_letters:
                        probs[i] = 1.0 / unguessed_count
        
        return probs
    
    def get_best_guess(self, masked_word, guessed_letters):
        """Get the most probable unguessed letter."""
        probs = self.predict_letter_probabilities(masked_word, guessed_letters)
        
        # Mask already guessed letters
        for letter in guessed_letters:
            if letter.isupper() and letter.isalpha():
                idx = letter_to_index(letter)
                if 0 <= idx < 26:
                    probs[idx] = 0
        
        if probs.sum() == 0:
            return None
        
        best_idx = np.argmax(probs)
        return index_to_letter(best_idx)
