"""
Advanced Contextual Hidden Markov Model for Hangman Letter Prediction
Implements per-length HMMs with 2-letter contextual conditioning and probability blending.
"""
import math
import pickle
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Optional


class ContextualHMM:
    """
    Hybrid probabilistic language model for Hangman.
    Estimates P(letter | position, context) using per-length HMMs with contextual emissions.
    """
    
    def __init__(self, corpus_path: str = "corpus.txt", alpha: float = 1.0):
        """
        Initialize the contextual HMM model.
        
        Args:
            corpus_path: Path to training corpus file
            alpha: Laplace smoothing parameter (default: 1.0)
        """
        self.corpus_path = corpus_path
        self.alpha = alpha
        self.letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        self.letter_set = set(self.letters)
        
        # Per-length models: models[length][position][(prev2, prev1)][letter] = count
        self.models: Dict[int, Dict[int, Dict[Tuple[str, str], Counter]]] = {}
        
        # Position-specific probabilities: pos_probs[length][position][letter] = prob
        self.pos_probs: Dict[int, Dict[int, Dict[str, float]]] = {}
        
        # Global letter probabilities: global_probs[letter] = prob
        self.global_probs: Dict[str, float] = {}
        
        # Word lists by length for candidate filtering
        self.words_by_length: Dict[int, List[str]] = defaultdict(list)
        
        # Blending weights for context backoff
        self.lambda_2 = 0.6   # Two-letter context weight
        self.lambda_1 = 0.2   # One-letter context weight
        self.lambda_pos = 0.15  # Position weight
        self.lambda_global = 0.05  # Global weight
        
        self.trained = False
    
    def train(self, corpus_file: Optional[str] = None, verbose: bool = True):
        """
        Build contextual HMMs for each word length with Laplace smoothing.
        
        Args:
            corpus_file: Path to corpus (optional, uses self.corpus_path if None)
            verbose: Whether to print training progress
        """
        if corpus_file:
            self.corpus_path = corpus_file
            
        if verbose:
            print("Training Contextual HMM model...")
        
        # Load and preprocess corpus
        words = self._load_corpus()
        
        if verbose:
            print(f"Loaded {len(words)} words from corpus")
        
        # Initialize counters
        global_counter = Counter()
        length_counters = defaultdict(lambda: defaultdict(Counter))  # [length][pos][letter]
        context_counters = defaultdict(lambda: defaultdict(lambda: defaultdict(Counter)))  # [length][pos][(p2,p1)][letter]
        
        # Count occurrences
        for word in words:
            length = len(word)
            self.words_by_length[length].append(word)
            
            for pos, letter in enumerate(word):
                # Global counts
                global_counter[letter] += 1
                
                # Position counts
                length_counters[length][pos][letter] += 1
                
                # Contextual counts with 2-letter history
                prev1 = word[pos - 1] if pos >= 1 else "_"
                prev2 = word[pos - 2] if pos >= 2 else "_"
                context_counters[length][pos][(prev2, prev1)][letter] += 1
        
        # Build probability distributions with Laplace smoothing
        self.global_probs = self._compute_probs(global_counter, self.alpha)
        
        # Build per-length models
        for length in sorted(length_counters.keys()):
            self.models[length] = {}
            self.pos_probs[length] = {}
            
            for pos in range(length):
                # Position-specific probabilities
                self.pos_probs[length][pos] = self._compute_probs(
                    length_counters[length][pos], 
                    self.alpha
                )
                
                # Contextual emission probabilities
                self.models[length][pos] = {}
                for context, counter in context_counters[length][pos].items():
                    self.models[length][pos][context] = self._compute_probs(counter, self.alpha)
        
        self.trained = True
        
        if verbose:
            print(f"Trained {len(self.models)} length-based models")
            self._print_training_summary()
    
    def _load_corpus(self) -> List[str]:
        """Load and validate corpus words."""
        words = []
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip().upper()
                # Only keep valid alphabetic words
                if word and word.isalpha() and all(c in self.letter_set for c in word):
                    words.append(word)
        return words
    
    def _compute_probs(self, counter: Counter, alpha: float) -> Dict[str, float]:
        """
        Convert counter to smoothed probability distribution.
        
        Args:
            counter: Counter object with letter counts
            alpha: Smoothing parameter
        
        Returns:
            Dictionary mapping letters to probabilities
        """
        total = sum(counter.values()) + alpha * len(self.letters)
        probs = {}
        
        for letter in self.letters:
            probs[letter] = (counter.get(letter, 0) + alpha) / total
        
        return probs
    
    def _get_context_probs(self, length: int, pos: int, prev2: str, prev1: str) -> Dict[str, float]:
        """
        Get probability distribution with context backoff.
        
        Blends probabilities from:
        1. Two-letter context: P(letter | prev2, prev1, pos)
        2. One-letter context: P(letter | prev1, pos)
        3. Position: P(letter | pos)
        4. Global: P(letter)
        
        Args:
            length: Word length
            pos: Position in word
            prev2: Second previous letter (or "_")
            prev1: Previous letter (or "_")
        
        Returns:
            Blended probability distribution
        """
        probs = {letter: 0.0 for letter in self.letters}
        
        # Check if we have this length model
        if length not in self.models or pos not in self.models[length]:
            # Fallback to global
            return self.global_probs.copy()
        
        # Get context probabilities
        ctx2_probs = self.models[length][pos].get((prev2, prev1), None)
        
        # One-letter context fallback
        ctx1_probs = None
        if ctx2_probs is None:
            # Try one-letter context
            for context, prob_dict in self.models[length][pos].items():
                if context[1] == prev1:  # Match prev1 only
                    ctx1_probs = prob_dict
                    break
        
        # Get position and global probs
        pos_probs = self.pos_probs[length][pos]
        global_probs = self.global_probs
        
        # Blend probabilities
        for letter in self.letters:
            p = 0.0
            
            if ctx2_probs is not None:
                # Full context available
                p += self.lambda_2 * ctx2_probs.get(letter, 0.0)
                p += self.lambda_pos * pos_probs.get(letter, 0.0)
                p += self.lambda_global * global_probs.get(letter, 0.0)
            elif ctx1_probs is not None:
                # One-letter context available
                p += (self.lambda_2 + self.lambda_1) * ctx1_probs.get(letter, 0.0)
                p += self.lambda_pos * pos_probs.get(letter, 0.0)
                p += self.lambda_global * global_probs.get(letter, 0.0)
            else:
                # Position and global only
                p += (self.lambda_2 + self.lambda_1 + self.lambda_pos) * pos_probs.get(letter, 0.0)
                p += self.lambda_global * global_probs.get(letter, 0.0)
            
            probs[letter] = p
        
        # Normalize
        total = sum(probs.values())
        if total > 0:
            probs = {k: v / total for k, v in probs.items()}
        
        return probs
    
    def predict_letter_probabilities(self, masked_word: str, guessed_letters: Set[str]) -> List[float]:
        """
        Predict probability distribution over unguessed letters.
        
        Main API for Hangman RL agent. Uses contextual HMM with candidate filtering.
        
        Args:
            masked_word: Current pattern (e.g., '_A__E')
            guessed_letters: Set of already guessed letters
        
        Returns:
            Normalized probability vector of size 26 (A-Z)
        """
        if not self.trained:
            raise RuntimeError("Model must be trained before prediction")
        
        length = len(masked_word)
        masked_word = masked_word.upper()
        guessed_letters = {g.upper() for g in guessed_letters}
        
        # Get candidate words matching pattern
        candidates = self._filter_candidates(masked_word, guessed_letters, length)
        
        if not candidates:
            # No candidates, use global distribution
            probs = [self.global_probs.get(letter, 0.0) for letter in self.letters]
        else:
            # Weight letters by candidate word probabilities
            letter_scores = Counter()
            
            for word in candidates:
                # Compute word probability under HMM
                word_prob = self._score_word(word)
                
                # Count letters in this word (not yet guessed)
                for letter in set(word):
                    if letter not in guessed_letters:
                        letter_scores[letter] += word_prob
            
            # Convert to probabilities with smoothing
            probs_dict = self._compute_probs(letter_scores, self.alpha * 0.1)  # Light smoothing
            probs = [probs_dict.get(letter, 0.0) for letter in self.letters]
        
        # Additionally blend with position-based context for blank positions
        if length in self.models:
            context_probs = self._get_pattern_context_probs(masked_word, length)
            # Blend: 70% candidate-based, 30% context-based
            probs = [0.7 * p + 0.3 * context_probs[i] for i, p in enumerate(probs)]
        
        # Mask already guessed letters
        for letter in guessed_letters:
            if letter in self.letter_set:
                idx = self.letters.index(letter)
                probs[idx] = 0.0
        
        # Normalize
        total = sum(probs)
        if total > 0:
            probs = [p / total for p in probs]
        else:
            # Uniform over unguessed
            unguessed = [1.0 if self.letters[i] not in guessed_letters else 0.0 for i in range(26)]
            total = sum(unguessed)
            probs = [u / total if total > 0 else 0.0 for u in unguessed]
        
        return probs
    
    def _filter_candidates(self, pattern: str, guessed_letters: Set[str], length: int) -> List[str]:
        """
        Filter candidate words matching the pattern.
        
        Args:
            pattern: Masked word pattern
            guessed_letters: Set of guessed letters
            length: Word length
        
        Returns:
            List of matching words
        """
        candidates = []
        correct_letters = set(c for c in pattern if c != '_')
        wrong_letters = guessed_letters - correct_letters
        
        for word in self.words_by_length.get(length, []):
            # Skip if contains wrong letters
            if any(letter in word for letter in wrong_letters):
                continue
            
            # Check pattern match
            match = True
            for i, c in enumerate(pattern):
                if c != '_' and word[i] != c:
                    match = False
                    break
                elif c == '_' and word[i] in correct_letters:
                    match = False
                    break
            
            if match:
                candidates.append(word)
        
        return candidates
    
    def _score_word(self, word: str) -> float:
        """
        Compute probability of word under contextual HMM.
        
        Args:
            word: Word to score
        
        Returns:
            Log-probability score
        """
        length = len(word)
        log_prob = 0.0
        
        if length not in self.models:
            return 1e-10  # Very small prob for unseen length
        
        for pos, letter in enumerate(word):
            prev1 = word[pos - 1] if pos >= 1 else "_"
            prev2 = word[pos - 2] if pos >= 2 else "_"
            
            # Get contextual probability
            probs = self._get_context_probs(length, pos, prev2, prev1)
            p = probs.get(letter, 1e-10)
            log_prob += math.log(max(p, 1e-10))
        
        # Return exp(log_prob) for linear weighting
        return math.exp(log_prob)
    
    def _get_pattern_context_probs(self, pattern: str, length: int) -> List[float]:
        """
        Get probability distribution based on pattern context at blank positions.
        
        Args:
            pattern: Masked word pattern
            length: Word length
        
        Returns:
            Probability distribution over 26 letters
        """
        letter_probs = Counter()
        blank_count = 0
        
        for pos, c in enumerate(pattern):
            if c == '_':
                blank_count += 1
                prev1 = pattern[pos - 1] if pos >= 1 else "_"
                prev2 = pattern[pos - 2] if pos >= 2 else "_"
                
                # Get contextual probabilities for this position
                probs = self._get_context_probs(length, pos, prev2, prev1)
                
                for letter, prob in probs.items():
                    letter_probs[letter] += prob
        
        # Average over blank positions
        if blank_count > 0:
            letter_probs = {k: v / blank_count for k, v in letter_probs.items()}
        
        # Convert to list
        result = [letter_probs.get(letter, 0.0) for letter in self.letters]
        
        # Normalize
        total = sum(result)
        if total > 0:
            result = [r / total for r in result]
        
        return result
    
    def entropy(self, probs: List[float]) -> float:
        """
        Compute Shannon entropy of probability distribution.
        
        Args:
            probs: Probability distribution
        
        Returns:
            Entropy in bits
        """
        entropy = 0.0
        for p in probs:
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy
    
    def get_best_guess(self, masked_word: str, guessed_letters: Set[str]) -> Optional[str]:
        """
        Get the most probable unguessed letter.
        
        Args:
            masked_word: Current pattern
            guessed_letters: Set of guessed letters
        
        Returns:
            Best letter to guess, or None if no options
        """
        probs = self.predict_letter_probabilities(masked_word, guessed_letters)
        
        max_prob = max(probs)
        if max_prob == 0:
            return None
        
        best_idx = probs.index(max_prob)
        return self.letters[best_idx]
    
    def save(self, path: str = "hmm_model.pkl"):
        """
        Save model to file.
        
        Args:
            path: Path to save file
        """
        data = {
            'alpha': self.alpha,
            'letters': self.letters,
            'models': self.models,
            'pos_probs': self.pos_probs,
            'global_probs': self.global_probs,
            'words_by_length': self.words_by_length,
            'lambda_2': self.lambda_2,
            'lambda_1': self.lambda_1,
            'lambda_pos': self.lambda_pos,
            'lambda_global': self.lambda_global,
            'trained': self.trained
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Model saved to {path}")
    
    def load(self, path: str = "hmm_model.pkl"):
        """
        Load model from file.
        
        Args:
            path: Path to load file
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.alpha = data['alpha']
        self.letters = data['letters']
        self.letter_set = set(self.letters)
        self.models = data['models']
        self.pos_probs = data['pos_probs']
        self.global_probs = data['global_probs']
        self.words_by_length = data['words_by_length']
        self.lambda_2 = data['lambda_2']
        self.lambda_1 = data['lambda_1']
        self.lambda_pos = data['lambda_pos']
        self.lambda_global = data['lambda_global']
        self.trained = data['trained']
        
        print(f"Model loaded from {path}")
    
    def _print_training_summary(self):
        """Print summary statistics of trained model."""
        print(f"  Word lengths covered: {sorted(self.models.keys())}")
        
        # Sample statistics
        if self.models:
            sample_length = sorted(self.models.keys())[len(self.models) // 2]
            if sample_length in self.models and 2 in self.models[sample_length]:
                contexts = list(self.models[sample_length][2].keys())
                if contexts:
                    sample_ctx = contexts[0]
                    sample_probs = self.models[sample_length][2][sample_ctx]
                    top_letters = sorted(sample_probs.items(), key=lambda x: x[1], reverse=True)[:3]
                    top_letters_str = ','.join([f"{l}({p:.3f})" for l, p in top_letters])
                    print(f"  Example: length={sample_length} pos=2 ctx={sample_ctx} → {top_letters_str}")
        
        print(f"  Global top-5 letters: {self._get_top_letters(self.global_probs, 5)}")
    
    def _get_top_letters(self, probs: Dict[str, float], n: int = 5) -> str:
        """Get top N letters by probability."""
        sorted_letters = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:n]
        return ','.join([f"{l}({p:.3f})" for l, p in sorted_letters])


# Alias for backward compatibility
HangmanHMM = ContextualHMM
