"""
Hangman Game Environment (OpenAI Gym-style)
"""
import random
from utils import get_masked_word


class HangmanEnv:
    """
    Hangman game environment for reinforcement learning.
    """
    
    def __init__(self, words, max_wrong_guesses=6):
        """
        Initialize Hangman environment.
        
        Args:
            words: List of words to use for games
            max_wrong_guesses: Maximum wrong guesses allowed
        """
        self.words = words
        self.max_wrong_guesses = max_wrong_guesses
        self.reset()
    
    def reset(self, word=None):
        """
        Reset the environment for a new game.
        
        Args:
            word: Specific word to use (optional, otherwise random)
        
        Returns:
            Initial state
        """
        if word is not None:
            self.target_word = word.upper()
        else:
            self.target_word = random.choice(self.words).upper()
        
        self.guessed_letters = set()
        self.correct_guesses = set()
        self.wrong_guesses = set()
        self.lives = self.max_wrong_guesses
        self.done = False
        
        return self._get_state()
    
    def step(self, action):
        """
        Take an action (guess a letter).
        
        Args:
            action: Letter to guess (A-Z)
        
        Returns:
            (state, reward, done, info)
        """
        if self.done:
            raise ValueError("Game is already finished. Call reset() first.")
        
        action = action.upper()
        
        # Check for repeated guess
        if action in self.guessed_letters:
            reward = -2
            info = {'repeated': True, 'correct': False}
            return self._get_state(), reward, self.done, info
        
        self.guessed_letters.add(action)
        
        # Check if guess is correct
        if action in self.target_word:
            self.correct_guesses.add(action)
            reward = 10
            info = {'repeated': False, 'correct': True}
            
            # Check if word is complete
            if all(letter in self.correct_guesses for letter in self.target_word):
                self.done = True
                reward = 100  # Win bonus
                info['won'] = True
        else:
            self.wrong_guesses.add(action)
            self.lives -= 1
            reward = -5
            info = {'repeated': False, 'correct': False}
            
            # Check if out of lives
            if self.lives <= 0:
                self.done = True
                reward = -50  # Loss penalty
                info['won'] = False
        
        return self._get_state(), reward, self.done, info
    
    def _get_state(self):
        """Get current game state."""
        masked_word = get_masked_word(self.target_word, self.correct_guesses)
        return {
            'masked_word': masked_word,
            'guessed_letters': self.guessed_letters.copy(),
            'lives': self.lives,
            'target_word': self.target_word,
            'done': self.done
        }
    
    def render(self):
        """Display current game state."""
        state = self._get_state()
        print(f"\nWord: {state['masked_word']}")
        print(f"Lives: {state['lives']}")
        print(f"Guessed: {sorted(state['guessed_letters'])}")
        if self.done:
            print(f"Target: {self.target_word}")
            if state['lives'] > 0:
                print("YOU WON!")
            else:
                print("YOU LOST!")
    
    def get_available_actions(self):
        """Get list of letters that haven't been guessed yet."""
        all_letters = set(chr(i) for i in range(ord('A'), ord('Z') + 1))
        return list(all_letters - self.guessed_letters)
