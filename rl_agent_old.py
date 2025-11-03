"""
Reinforcement Learning Agent for Hangman
"""
import numpy as np
import random
from collections import defaultdict
from tqdm import tqdm
from utils import letter_to_index, index_to_letter


class QLearningAgent:
    """
    Q-Learning agent for Hangman game.
    Uses HMM probabilities as features for action selection.
    """
    
    def __init__(self, hmm_model, alpha=0.1, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        """
        Initialize Q-Learning agent.
        
        Args:
            hmm_model: Trained HMM model for letter probabilities
            alpha: Learning rate
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_decay: Decay rate for epsilon
            epsilon_min: Minimum epsilon value
        """
        self.hmm_model = hmm_model
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = defaultdict(lambda: defaultdict(float))
    
    def get_state_key(self, state):
        """Convert state to a hashable key for Q-table."""
        masked_word = state['masked_word']
        lives = state['lives']
        guessed = frozenset(state['guessed_letters'])
        return (masked_word, lives, guessed)
    
    def choose_action(self, state, available_actions, training=True):
        """
        Choose an action using epsilon-greedy policy with HMM guidance.
        
        Args:
            state: Current game state
            available_actions: List of unguessed letters
            training: Whether in training mode
        
        Returns:
            Selected letter
        """
        if not available_actions:
            return None
        
        # Epsilon-greedy exploration
        if training and random.random() < self.epsilon:
            # Explore: use HMM probabilities for smarter exploration
            hmm_probs = self.hmm_model.predict_letter_probabilities(
                state['masked_word'], 
                state['guessed_letters']
            )
            
            # Sample from HMM distribution
            available_probs = []
            for letter in available_actions:
                idx = letter_to_index(letter)
                if 0 <= idx < 26:
                    available_probs.append(hmm_probs[idx])
                else:
                    available_probs.append(0)
            
            # Normalize
            total = sum(available_probs)
            if total > 0:
                available_probs = [p / total for p in available_probs]
                action = np.random.choice(available_actions, p=available_probs)
            else:
                action = random.choice(available_actions)
        else:
            # Exploit: choose action with highest Q-value + HMM guidance
            state_key = self.get_state_key(state)
            hmm_probs = self.hmm_model.predict_letter_probabilities(
                state['masked_word'], 
                state['guessed_letters']
            )
            
            best_value = float('-inf')
            best_action = None
            
            for letter in available_actions:
                idx = letter_to_index(letter)
                # Combine Q-value with HMM probability
                q_value = self.q_table[state_key][letter]
                if 0 <= idx < 26:
                    hmm_bonus = hmm_probs[idx] * 10  # Scale HMM contribution
                else:
                    hmm_bonus = 0
                combined_value = q_value + hmm_bonus
                
                if combined_value > best_value:
                    best_value = combined_value
                    best_action = letter
            
            action = best_action if best_action else random.choice(available_actions)
        
        return action
    
    def update(self, state, action, reward, next_state, done):
        """
        Update Q-table using Q-learning update rule.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        # Current Q-value
        current_q = self.q_table[state_key][action]
        
        # Maximum Q-value for next state
        if done:
            max_next_q = 0
        else:
            next_q_values = self.q_table[next_state_key]
            max_next_q = max(next_q_values.values()) if next_q_values else 0
        
        # Q-learning update
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state_key][action] = new_q
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def train(self, env, num_episodes=5000, verbose=True):
        """
        Train the agent on the Hangman environment.
        
        Args:
            env: HangmanEnv instance
            num_episodes: Number of training episodes
            verbose: Whether to print progress
        
        Returns:
            List of rewards per episode
        """
        episode_rewards = []
        episode_wins = []
        
        # Progress bar
        pbar = tqdm(range(num_episodes), desc="Training Q-Learning Agent", ncols=100)
        
        for episode in pbar:
            state = env.reset()
            total_reward = 0
            
            while not state['done']:
                # Choose action
                available_actions = env.get_available_actions()
                action = self.choose_action(state, available_actions, training=True)
                
                # Take step
                next_state, reward, done, info = env.step(action)
                total_reward += reward
                
                # Update Q-table
                self.update(state, action, reward, next_state, done)
                
                state = next_state
            
            # Track win
            won = 1 if state['lives'] > 0 else 0
            episode_wins.append(won)
            episode_rewards.append(total_reward)
            self.decay_epsilon()
            
            # Update progress bar
            if episode >= 99:  # Start showing stats after 100 episodes
                avg_reward = np.mean(episode_rewards[-100:])
                win_rate = np.mean(episode_wins[-100:]) * 100
                pbar.set_postfix({
                    'Reward': f'{avg_reward:.1f}',
                    'Win%': f'{win_rate:.1f}',
                    'ε': f'{self.epsilon:.3f}'
                })
        
        pbar.close()
        return episode_rewards
