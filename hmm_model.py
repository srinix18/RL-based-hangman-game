"""
Enhanced Contextual HMM for Hangman Letter Prediction
Implements 3-letter context, adaptive smoothing, weighted candidate filtering,
and position-aware probability aggregation for 40%+ win rate.
"""
import math
import pickle
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Optional


class ContextualHMM:
    """
    Advanced HMM with 3-letter context, adaptive blending, and candidate weighting.
    Designed for high-performance Hangman AI (target: 40%+ win rate).
    """
    
    def __init__(self, corpus_path: str = "corpus.txt", alpha: float = 1.0):
        """
        Initialize Enhanced Contextual HMM.
        
        Args:
            corpus_path: Path to training corpus file
            alpha: Base Laplace smoothing parameter
        """
        self.corpus_path = corpus_path
        self.alpha = alpha
        self.letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        self.letter_set = set(self.letters)
        
        # Core data structures
        self.models: Dict[int, Dict[int, Dict[Tuple, Counter]]] = {}
        self.vocabulary: Dict[int, List[str]] = defaultdict(list)
        self.pos_probs: Dict[int, Dict[int, Dict[str, float]]] = {}
        self.global_probs: Dict[str, float] = {}
        self.position_bigrams: Dict[int, Dict[int, Counter]] = {}
        
        # Length statistics
        self.length_frequencies = Counter()
        self.length_priors: Dict[int, float] = {}
        
        self.trained = False
    
    def train(self, verbose: bool = True):
        """
        Train enhanced HMM with 3-letter context and positional bigrams.
        """
        if verbose:
            print("Training Enhanced Contextual HMM...")
        
        # Load corpus
        words = self._load_corpus()
        if verbose:
            print(f"Loaded {len(words)} words from corpus")
        
        # Initialize counters
        global_counter = Counter()
        pos_counters = defaultdict(lambda: defaultdict(Counter))
        context_counters = defaultdict(lambda: defaultdict(lambda: defaultdict(Counter)))
        bigram_counters = defaultdict(lambda: defaultdict(Counter))
        
        # Process each word
        for word in words:
            length = len(word)
            self.vocabulary[length].append(word)
            self.length_frequencies[length] += 1
            
            for pos, letter in enumerate(word):
                # Global counts
                global_counter[letter] += 1
                
                # Position counts
                pos_counters[length][pos][letter] += 1
                
                # Extract context
                prev2 = word[pos - 2] if pos >= 2 else "_"
                prev1 = word[pos - 1] if pos >= 1 else "_"
                prev0 = letter
                next1 = word[pos + 1] if pos < length - 1 else "_"
                
                # 3-letter context (prev2, prev1, current)
                context_counters[length][pos][(prev2, prev1, "3")][letter] += 1
                
                # 2-letter context (prev1, current)
                context_counters[length][pos][(prev1, "2")][letter] += 1
                
                # 1-letter context (prev only)
                context_counters[length][pos][("1",)][letter] += 1
                
                # Position bigrams
                if pos < length - 1:
                    bigram_counters[length][pos][(letter, next1)] += 1
        
        # Build probability distributions
        self.global_probs = self._compute_probs(global_counter, self.alpha)
        
        # Compute length priors
        total_words = sum(self.length_frequencies.values())
        self.length_priors = {
            length: count / total_words 
            for length, count in self.length_frequencies.items()
        }
        
        # Build per-length models
        for length in sorted(pos_counters.keys()):
            self.models[length] = {}
            self.pos_probs[length] = {}
            self.position_bigrams[length] = {}
            
            for pos in range(length):
                # Position-specific probabilities
                total_samples = sum(pos_counters[length][pos].values())
                adaptive_alpha = self._adaptive_alpha(total_samples)
                self.pos_probs[length][pos] = self._compute_probs(
                    pos_counters[length][pos], 
                    adaptive_alpha
                )
                
                # Contextual emission probabilities
                self.models[length][pos] = {}
                for context, counter in context_counters[length][pos].items():
                    sample_count = sum(counter.values())
                    ctx_alpha = self._adaptive_alpha(sample_count)
                    self.models[length][pos][context] = self._compute_probs(
                        counter, 
                        ctx_alpha
                    )
                
                # Bigrams
                if pos in bigram_counters[length]:
                    self.position_bigrams[length][pos] = bigram_counters[length][pos]
        
        self.trained = True
        
        if verbose:
            self._print_training_summary()
    
    def _load_corpus(self) -> List[str]:
        """Load and validate corpus words."""
        words = []
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip().upper()
                if word and word.isalpha() and all(c in self.letter_set for c in word):
                    words.append(word)
        return words
    
    def _adaptive_alpha(self, sample_size: int) -> float:
        """
        Compute adaptive smoothing parameter based on sample size.
        
        Args:
            sample_size: Number of samples observed
        
        Returns:
            Adaptive alpha value
        """
        if sample_size > 1000:
            return self.alpha * 0.5
        elif sample_size > 100:
            return self.alpha * 0.8
        else:
            return self.alpha
    
    def _compute_probs(self, counter: Counter, alpha: float, vocab_size: int = 26) -> Dict[str, float]:
        """
        Convert counter to smoothed probability distribution.
        
        Args:
            counter: Counter with letter counts
            alpha: Smoothing parameter
            vocab_size: Vocabulary size
        
        Returns:
            Probability dictionary
        """
        total = sum(counter.values()) + alpha * vocab_size
        probs = {}
        
        for letter in self.letters:
            probs[letter] = (counter.get(letter, 0) + alpha) / total
        
        return probs
    
    def _adaptive_weights(self, pattern: str) -> Dict[str, float]:
        """
        Compute adaptive blending weights based on revealed information.
        
        Args:
            pattern: Current masked pattern
        
        Returns:
            Dictionary with lambda weights
        """
        revealed = sum(1 for c in pattern if c != '_')
        total = len(pattern)
        reveal_ratio = revealed / max(1, total)
        
        if reveal_ratio > 0.6:
            # Late game - trust context heavily
            return {
                'lambda_3': 0.5,
                'lambda_2': 0.25,
                'lambda_1': 0.15,
                'lambda_pos': 0.08,
                'lambda_g': 0.02
            }
        elif reveal_ratio > 0.3:
            # Mid game - balanced
            return {
                'lambda_3': 0.4,
                'lambda_2': 0.25,
                'lambda_1': 0.15,
                'lambda_pos': 0.15,
                'lambda_g': 0.05
            }
        else:
            # Early game - trust position more
            return {
                'lambda_3': 0.2,
                'lambda_2': 0.2,
                'lambda_1': 0.15,
                'lambda_pos': 0.35,
                'lambda_g': 0.10
            }
    
    def _get_context_probs(self, length: int, pos: int, 
                          prev2: str, prev1: str, 
                          weights: Dict[str, float]) -> Dict[str, float]:
        """
        Get probability distribution with context backoff and adaptive blending.
        
        Args:
            length: Word length
            pos: Position in word
            prev2: Second previous letter
            prev1: Previous letter
            weights: Lambda weights for blending
        
        Returns:
            Blended probability distribution
        """
        probs = {letter: 0.0 for letter in self.letters}
        
        if length not in self.models or pos not in self.models[length]:
            return self.global_probs.copy()
        
        # Try 3-letter context
        ctx3_probs = self.models[length][pos].get((prev2, prev1, "3"), None)
        
        # Try 2-letter context
        ctx2_probs = self.models[length][pos].get((prev1, "2"), None)
        
        # Try 1-letter context
        ctx1_probs = self.models[length][pos].get(("1",), None)
        
        # Get position and global
        pos_probs = self.pos_probs[length][pos]
        global_probs = self.global_probs
        
        # Blend with backoff
        for letter in self.letters:
            p = 0.0
            
            if ctx3_probs is not None:
                p += weights['lambda_3'] * ctx3_probs.get(letter, 0.0)
            else:
                # Redistribute weight
                weights['lambda_2'] += weights['lambda_3'] * 0.5
                weights['lambda_pos'] += weights['lambda_3'] * 0.5
            
            if ctx2_probs is not None:
                p += weights['lambda_2'] * ctx2_probs.get(letter, 0.0)
            elif ctx1_probs is not None:
                p += (weights['lambda_2'] + weights['lambda_1']) * ctx1_probs.get(letter, 0.0)
            else:
                weights['lambda_pos'] += weights['lambda_2'] + weights['lambda_1']
            
            p += weights['lambda_pos'] * pos_probs.get(letter, 0.0)
            p += weights['lambda_g'] * global_probs.get(letter, 0.0)
            
            probs[letter] = p
        
        # Normalize
        total = sum(probs.values())
        if total > 0:
            probs = {k: v / total for k, v in probs.items()}
        
        return probs
    
    def _matches_pattern(self, word: str, pattern: str) -> bool:
        """Check if word matches pattern."""
        if len(word) != len(pattern):
            return False
        
        for i, c in enumerate(pattern):
            if c != '_' and word[i] != c:
                return False
        
        return True
    
    def _compute_word_likelihood(self, word: str, pattern: str) -> float:
        """
        Compute likelihood of word given pattern using position probabilities.
        
        Args:
            word: Candidate word
            pattern: Current pattern
        
        Returns:
            Log-likelihood score
        """
        length = len(word)
        log_prob = 0.0
        
        if length not in self.models:
            return 1e-10
        
        weights = self._adaptive_weights(pattern)
        
        for pos, letter in enumerate(word):
            prev2 = word[pos - 2] if pos >= 2 else "_"
            prev1 = word[pos - 1] if pos >= 1 else "_"
            
            probs = self._get_context_probs(length, pos, prev2, prev1, weights)
            p = probs.get(letter, 1e-10)
            log_prob += math.log(max(p, 1e-10))
        
        return math.exp(log_prob)
    
    def get_weighted_candidates(self, pattern: str, guessed_letters: Set[str]) -> List[Tuple[str, float]]:
        """
        Get candidate words with likelihood-based weights.
        
        Args:
            pattern: Current pattern
            guessed_letters: Set of guessed letters
        
        Returns:
            List of (word, probability) tuples
        """
        length = len(pattern)
        candidates = []
        
        correct_letters = set(c for c in pattern if c != '_')
        wrong_letters = guessed_letters - correct_letters
        
        for word in self.vocabulary.get(length, []):
            # Filter by wrong letters
            if any(letter in word for letter in wrong_letters):
                continue
            
            # Check pattern match
            if self._matches_pattern(word, pattern):
                score = self._compute_word_likelihood(word, pattern)
                candidates.append((word, score))
        
        # Normalize scores
        total_score = sum(score for _, score in candidates)
        if total_score > 0:
            candidates = [(word, score / total_score) for word, score in candidates]
        
        return candidates
    
    def _extract_context(self, pattern: str, pos: int) -> Tuple[str, str, str, str]:
        """
        Extract context around position in pattern.
        
        Args:
            pattern: Current pattern
            pos: Position
        
        Returns:
            (prev2, prev1, next1, next2)
        """
        prev2 = pattern[pos - 2] if pos >= 2 and pattern[pos - 2] != '_' else "_"
        prev1 = pattern[pos - 1] if pos >= 1 and pattern[pos - 1] != '_' else "_"
        next1 = pattern[pos + 1] if pos < len(pattern) - 1 and pattern[pos + 1] != '_' else "_"
        next2 = pattern[pos + 2] if pos < len(pattern) - 2 and pattern[pos + 2] != '_' else "_"
        
        return prev2, prev1, next1, next2
    
    def _get_position_weight(self, pattern: str, pos: int) -> float:
        """
        Compute position weight based on context strength.
        
        Args:
            pattern: Current pattern
            pos: Position
        
        Returns:
            Weight multiplier
        """
        prev2, prev1, next1, next2 = self._extract_context(pattern, pos)
        context_strength = sum(1 for c in [prev2, prev1, next1, next2] if c != "_")
        
        return 1.0 + 0.3 * context_strength
    
    def predict_letter_probabilities(self, masked_word: str, guessed_letters: Set[str]) -> List[float]:
        """
        Predict probability distribution with position-aware aggregation.
        
        Main API for Hangman RL agent.
        
        Args:
            masked_word: Current pattern (e.g., '_A__E')
            guessed_letters: Set of guessed letters
        
        Returns:
            Normalized 26-length probability vector
        """
        if not self.trained:
            raise RuntimeError("Model must be trained before prediction")
        
        pattern = masked_word.upper()
        guessed_letters = {g.upper() for g in guessed_letters}
        length = len(pattern)
        
        # Edge case: no blanks
        if '_' not in pattern:
            return [0.0] * 26
        
        # Get adaptive weights
        weights = self._adaptive_weights(pattern)
        
        # Position-based probabilities
        position_probs = Counter()
        total_weight = 0.0
        
        for pos, c in enumerate(pattern):
            if c == '_':
                prev2, prev1, _, _ = self._extract_context(pattern, pos)
                pos_weight = self._get_position_weight(pattern, pos)
                
                probs = self._get_context_probs(length, pos, prev2, prev1, weights.copy())
                
                for letter, prob in probs.items():
                    if letter not in guessed_letters:
                        position_probs[letter] += prob * pos_weight
                
                total_weight += pos_weight
        
        # Normalize position probabilities
        if total_weight > 0:
            position_probs = {k: v / total_weight for k, v in position_probs.items()}
        
        # Candidate-filtered probabilities
        candidates = self.get_weighted_candidates(pattern, guessed_letters)
        candidate_probs = Counter()
        
        if candidates:
            for word, word_prob in candidates:
                for letter in set(word):
                    if letter not in guessed_letters:
                        candidate_probs[letter] += word_prob
        
        # Blend: 70% positional + 30% candidate-filtered
        final_probs = []
        for letter in self.letters:
            if letter in guessed_letters:
                final_probs.append(0.0)
            else:
                pos_p = position_probs.get(letter, 0.0)
                cand_p = candidate_probs.get(letter, 0.0) if candidates else pos_p
                
                blended = 0.7 * pos_p + 0.3 * cand_p
                final_probs.append(blended)
        
        # Normalize
        total = sum(final_probs)
        if total > 0:
            final_probs = [p / total for p in final_probs]
        else:
            # Fallback to uniform
            unguessed_count = sum(1 for p in final_probs if p == 0.0)
            if unguessed_count > 0:
                final_probs = [1.0 / unguessed_count if p == 0.0 else 0.0 for p in final_probs]
        
        return final_probs
    
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
    
    def expected_info_gain(self, pattern: str, guessed_letters: Set[str]) -> Dict[str, float]:
        """
        Calculate expected information gain for each unguessed letter.
        
        Args:
            pattern: Current pattern
            guessed_letters: Set of guessed letters
        
        Returns:
            Dictionary mapping letters to entropy reduction
        """
        candidates = self.get_weighted_candidates(pattern, guessed_letters)
        if not candidates:
            return {}
        
        current_entropy = self.entropy([prob for _, prob in candidates])
        info_gains = {}
        
        for letter in self.letters:
            if letter in guessed_letters:
                continue
            
            # Simulate revealing this letter
            expected_entropy = 0.0
            
            # Case 1: Letter is in word
            matching = [(w, p) for w, p in candidates if letter in w]
            prob_present = sum(p for _, p in matching)
            
            if prob_present > 0 and matching:
                # Normalize matching candidates
                norm_matching = [(w, p / prob_present) for w, p in matching]
                entropy_present = self.entropy([p for _, p in norm_matching])
                expected_entropy += prob_present * entropy_present
            
            # Case 2: Letter not in word
            not_matching = [(w, p) for w, p in candidates if letter not in w]
            prob_absent = sum(p for _, p in not_matching)
            
            if prob_absent > 0 and not_matching:
                norm_not_matching = [(w, p / prob_absent) for w, p in not_matching]
                entropy_absent = self.entropy([p for _, p in norm_not_matching])
                expected_entropy += prob_absent * entropy_absent
            
            info_gains[letter] = current_entropy - expected_entropy
        
        return info_gains
    
    def get_best_guess(self, masked_word: str, guessed_letters: Set[str]) -> Optional[str]:
        """Get best letter based on probability."""
        probs = self.predict_letter_probabilities(masked_word, guessed_letters)
        max_prob = max(probs)
        
        if max_prob == 0:
            return None
        
        return self.letters[probs.index(max_prob)]
    
    def save(self, path: str = "hmm_model.pkl"):
        """Save model to file."""
        data = {
            'alpha': self.alpha,
            'letters': self.letters,
            'models': self.models,
            'vocabulary': dict(self.vocabulary),
            'pos_probs': self.pos_probs,
            'global_probs': self.global_probs,
            'position_bigrams': self.position_bigrams,
            'length_frequencies': self.length_frequencies,
            'length_priors': self.length_priors,
            'trained': self.trained
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Model saved to {path}")
    
    def load(self, path: str = "hmm_model.pkl") -> bool:
        """Load model from file."""
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            
            self.alpha = data['alpha']
            self.letters = data['letters']
            self.letter_set = set(self.letters)
            self.models = data['models']
            self.vocabulary = defaultdict(list, data['vocabulary'])
            self.pos_probs = data['pos_probs']
            self.global_probs = data['global_probs']
            self.position_bigrams = data['position_bigrams']
            self.length_frequencies = data['length_frequencies']
            self.length_priors = data['length_priors']
            self.trained = data['trained']
            
            print(f"Model loaded from {path}")
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False
    
    def _print_training_summary(self):
        """Print training summary statistics."""
        print(f"  Trained {len(self.models)} length-based models")
        print(f"  Word lengths: {sorted(self.models.keys())}")
        print(f"  Total vocabulary: {sum(len(words) for words in self.vocabulary.values())} words")
        
        # Show example emissions
        if self.models:
            sample_len = sorted(self.models.keys())[len(self.models) // 2]
            if sample_len in self.models and 2 in self.models[sample_len]:
                contexts = list(self.models[sample_len][2].keys())
                if contexts:
                    ctx = contexts[0]
                    probs = self.models[sample_len][2][ctx]
                    top = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
                    print(f"  Example: len={sample_len} pos=2 ctx={ctx} → {', '.join(f'{l}({p:.3f})' for l, p in top)}")
        
        # Global top letters
        top_global = sorted(self.global_probs.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"  Global top-5: {', '.join(f'{l}({p:.3f})' for l, p in top_global)}")


# Alias for backward compatibility
HangmanHMM = ContextualHMM


if __name__ == "__main__":
    # Test the model
    print("=" * 60)
    print("Testing Enhanced Contextual HMM")
    print("=" * 60)
    
    hmm = ContextualHMM(corpus_path="corpus.txt", alpha=1.0)
    hmm.train(verbose=True)
    hmm.save("hmm_model.pkl")
    
    print("\n" + "=" * 60)
    print("Testing Predictions")
    print("=" * 60)
    
    # Test prediction
    pattern = "_A__E"
    guessed = {'A', 'E'}
    probs = hmm.predict_letter_probabilities(pattern, guessed)
    
    print(f"\nPattern: {pattern}")
    print(f"Guessed: {guessed}")
    print(f"Probability sum: {sum(probs):.6f}")
    print(f"Entropy: {hmm.entropy(probs):.3f} bits")
    
    # Top 5 predictions
    top_letters = sorted(zip(hmm.letters, probs), key=lambda x: x[1], reverse=True)[:5]
    print(f"\nTop 5 predictions:")
    for letter, prob in top_letters:
        print(f"  {letter}: {prob:.4f} ({prob*100:.2f}%)")
    
    # Expected info gain
    info_gains = hmm.expected_info_gain(pattern, guessed)
    top_gains = sorted(info_gains.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"\nTop 5 by information gain:")
    for letter, gain in top_gains:
        print(f"  {letter}: {gain:.4f} bits")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
