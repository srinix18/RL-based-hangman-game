"""
GPU-Accelerated Q-Learning Agent for Hangman
Uses PyTorch tensors and CUDA for faster training
"""
import numpy as np
import random
import torch
import torch.nn.functional as F
from collections import defaultdict
from tqdm import tqdm
from utils import letter_to_index, index_to_letter


class GPUQLearningAgent:
    """
    Q-Learning agent with GPU acceleration using PyTorch.
    Uses tensor operations for faster Q-value updates.
    """
    
    def __init__(self, hmm_model, alpha=0.1, gamma=0.95, epsilon=1.0, 
                 epsilon_decay=0.995, epsilon_min=0.01, use_gpu=True):
        """
        Initialize GPU-accelerated Q-Learning agent.
        
        Args:
            hmm_model: Trained HMM model for letter probabilities
            alpha: Learning rate
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_decay: Decay rate for epsilon
            epsilon_min: Minimum epsilon value
            use_gpu: Whether to use GPU if available
        """
        self.hmm_model = hmm_model
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Set device (CUDA if available and requested)
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Version: {torch.version.cuda}")
        else:
            self.device = torch.device("cpu")
            if use_gpu:
                print("⚠ CUDA not available, using CPU")
            else:
                print("✓ Using CPU (GPU disabled)")
        
        # Q-table stored as nested dicts (state -> action -> Q-value)
        # But we'll batch-process updates using tensors
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        # Cache for batch updates
        self.update_cache = []
        self.cache_size = 32  # Process updates in batches
    
    def get_state_key(self, state):
        """Convert state to a hashable key for Q-table."""
        masked_word = state['masked_word']
        lives = state['lives']
        guessed = frozenset(state['guessed_letters'])
        return (masked_word, lives, guessed)
    
    def choose_action(self, state, available_actions, training=True, fast_mode=True):
        """
        Choose an action using epsilon-greedy policy with HMM guidance.
        
        Args:
            state: Current game state
            available_actions: List of unguessed letters
            training: Whether in training mode
            fast_mode: Use faster but slightly less accurate HMM predictions
        
        Returns:
            Selected letter
        """
        if not available_actions:
            return None
        
        # Get HMM probabilities (with optional caching for speed)
        if fast_mode and training:
            # Faster mode: skip expensive candidate scoring sometimes
            # Use cached or simplified predictions
            hmm_probs = self._fast_hmm_predict(state['masked_word'], state['guessed_letters'])
        else:
            # Full mode: use complete HMM prediction
            hmm_probs = self.hmm_model.predict_letter_probabilities(
                state['masked_word'], 
                state['guessed_letters']
            )
        
        # Convert to tensor and move to device
        hmm_probs_tensor = torch.tensor(hmm_probs, dtype=torch.float32, device=self.device)
        
        # Epsilon-greedy exploration
        if training and random.random() < self.epsilon:
            # Explore: sample from HMM probabilities
            available_probs = []
            for letter in available_actions:
                idx = letter_to_index(letter)
                if 0 <= idx < 26:
                    available_probs.append(hmm_probs[idx])
                else:
                    available_probs.append(0)
            
            total = sum(available_probs)
            if total > 0:
                available_probs = [p / total for p in available_probs]
                action = np.random.choice(available_actions, p=available_probs)
            else:
                action = random.choice(available_actions)
        else:
            # Exploit: choose action with highest Q-value + HMM guidance
            state_key = self.get_state_key(state)
            
            # Compute Q-values with HMM bonus on GPU
            q_values = []
            for letter in available_actions:
                idx = letter_to_index(letter)
                q_value = self.q_table[state_key][letter]
                
                if 0 <= idx < 26:
                    hmm_bonus = hmm_probs_tensor[idx].item() * 10
                else:
                    hmm_bonus = 0
                
                combined_value = q_value + hmm_bonus
                q_values.append(combined_value)
            
            # Find best action
            best_idx = np.argmax(q_values)
            action = available_actions[best_idx]
        
        return action
    
    def update(self, state, action, reward, next_state, done):
        """
        Cache update for batch processing.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.update_cache.append((state, action, reward, next_state, done))
        
        # Process batch when cache is full
        if len(self.update_cache) >= self.cache_size:
            self._process_batch_updates()
    
    def _process_batch_updates(self):
        """Process cached updates in batch using GPU tensors."""
        if not self.update_cache:
            return
        
        # Extract batch data
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for state, action, reward, next_state, done in self.update_cache:
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        
        # Convert rewards and dones to tensors
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_tensor = torch.tensor([1.0 if d else 0.0 for d in dones], 
                                    dtype=torch.float32, device=self.device)
        
        # Process each update
        current_qs = []
        max_next_qs = []
        
        for i in range(len(states)):
            state_key = self.get_state_key(states[i])
            next_state_key = self.get_state_key(next_states[i])
            
            # Current Q-value
            current_q = self.q_table[state_key][actions[i]]
            current_qs.append(current_q)
            
            # Max Q-value for next state
            if dones[i]:
                max_next_q = 0.0
            else:
                next_q_values = self.q_table[next_state_key]
                max_next_q = max(next_q_values.values()) if next_q_values else 0.0
            max_next_qs.append(max_next_q)
        
        # Convert to tensors
        current_qs_tensor = torch.tensor(current_qs, dtype=torch.float32, device=self.device)
        max_next_qs_tensor = torch.tensor(max_next_qs, dtype=torch.float32, device=self.device)
        
        # Compute targets: reward + gamma * max_next_q * (1 - done)
        targets = rewards_tensor + self.gamma * max_next_qs_tensor * (1.0 - dones_tensor)
        
        # Compute TD errors
        td_errors = targets - current_qs_tensor
        
        # Update Q-values: Q += alpha * td_error
        new_qs = current_qs_tensor + self.alpha * td_errors
        
        # Move back to CPU and update Q-table
        new_qs_cpu = new_qs.cpu().numpy()
        
        for i in range(len(states)):
            state_key = self.get_state_key(states[i])
            self.q_table[state_key][actions[i]] = float(new_qs_cpu[i])
        
        # Clear cache
        self.update_cache.clear()
    
    def _fast_hmm_predict(self, masked_word, guessed_letters):
        """
        Faster HMM prediction using simplified approach.
        Skips expensive candidate scoring for speed during training.
        
        Args:
            masked_word: Current pattern
            guessed_letters: Set of guessed letters
        
        Returns:
            Probability distribution (26-length array)
        """
        length = len(masked_word)
        
        # Use only position and global probs (skip expensive candidate filtering)
        if length in self.hmm_model.models:
            # Get context probs for blank positions
            probs = self.hmm_model._get_pattern_context_probs(masked_word.upper(), length)
        else:
            # Fallback to global
            probs = [self.hmm_model.global_probs.get(letter, 0.0) 
                    for letter in self.hmm_model.letters]
        
        # Mask guessed letters
        for letter in guessed_letters:
            if letter.upper() in self.hmm_model.letter_set:
                idx = self.hmm_model.letters.index(letter.upper())
                probs[idx] = 0.0
        
        # Normalize
        total = sum(probs)
        if total > 0:
            probs = [p / total for p in probs]
        else:
            # Uniform over unguessed
            unguessed = [1.0 if self.hmm_model.letters[i].upper() not in {g.upper() for g in guessed_letters} 
                        else 0.0 for i in range(26)]
            total = sum(unguessed)
            probs = [u / total if total > 0 else 0.0 for u in unguessed]
        
        return probs
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def train(self, env, num_episodes=5000, verbose=True, log_interval=100):
        """
        Train the agent on the Hangman environment with detailed per-episode metrics.
        
        Args:
            env: HangmanEnv instance
            num_episodes: Number of training episodes
            verbose: Whether to print progress
            log_interval: How often to log detailed metrics
        
        Returns:
            Tuple of (episode_rewards, episode_metrics)
        """
        episode_rewards = []
        episode_wins = []
        episode_metrics = []  # Detailed per-episode metrics
        
        # HMM prediction cache for speed
        hmm_cache = {}
        
        # Progress bar
        device_name = "GPU" if self.device.type == "cuda" else "CPU"
        pbar = tqdm(range(num_episodes), desc=f"Training Q-Learning on {device_name}", ncols=120)
        
        for episode in pbar:
            state = env.reset()
            total_reward = 0
            steps = 0
            correct_guesses = 0
            wrong_guesses = 0
            repeated_guesses = 0
            
            while not state['done']:
                # Choose action
                available_actions = env.get_available_actions()
                
                # Speed optimization: cache HMM predictions
                cache_key = (state['masked_word'], frozenset(state['guessed_letters']))
                if cache_key not in hmm_cache:
                    action = self.choose_action(state, available_actions, training=True)
                    # Cache the HMM probs for this state (optional)
                else:
                    action = self.choose_action(state, available_actions, training=True)
                
                # Take step
                next_state, reward, done, info = env.step(action)
                total_reward += reward
                steps += 1
                
                # Track per-episode stats
                if info.get('repeated', False):
                    repeated_guesses += 1
                elif info.get('correct', False):
                    correct_guesses += 1
                else:
                    wrong_guesses += 1
                
                # Update Q-table (cached for batch processing)
                self.update(state, action, reward, next_state, done)
                
                state = next_state
            
            # Process any remaining cached updates
            self._process_batch_updates()
            
            # Track metrics
            won = 1 if state['lives'] > 0 else 0
            episode_wins.append(won)
            episode_rewards.append(total_reward)
            
            # Store detailed per-episode metrics
            episode_metrics.append({
                'episode': episode,
                'reward': total_reward,
                'won': won,
                'steps': steps,
                'correct_guesses': correct_guesses,
                'wrong_guesses': wrong_guesses,
                'repeated_guesses': repeated_guesses,
                'lives_remaining': state['lives'],
                'word_length': len(state['masked_word']),
                'epsilon': self.epsilon
            })
            
            self.decay_epsilon()
            
            # Update progress bar
            if episode >= 99:  # Start showing stats after 100 episodes
                avg_reward = np.mean(episode_rewards[-100:])
                win_rate = np.mean(episode_wins[-100:]) * 100
                avg_wrong = np.mean([m['wrong_guesses'] for m in episode_metrics[-100:]])
                pbar.set_postfix({
                    'Reward': f'{avg_reward:.1f}',
                    'Win%': f'{win_rate:.1f}',
                    'AvgWrong': f'{avg_wrong:.1f}',
                    'ε': f'{self.epsilon:.3f}',
                    'States': len(self.q_table)
                })
            
            # Detailed logging at intervals
            if verbose and (episode + 1) % log_interval == 0 and episode >= 99:
                recent_metrics = episode_metrics[-log_interval:]
                avg_steps = np.mean([m['steps'] for m in recent_metrics])
                avg_correct = np.mean([m['correct_guesses'] for m in recent_metrics])
                avg_wrong = np.mean([m['wrong_guesses'] for m in recent_metrics])
                avg_repeated = np.mean([m['repeated_guesses'] for m in recent_metrics])
                win_rate_interval = np.mean([m['won'] for m in recent_metrics]) * 100
                
                print(f"\n[Episode {episode + 1}] Detailed Metrics (last {log_interval}):")
                print(f"  Win Rate: {win_rate_interval:.1f}%")
                print(f"  Avg Steps: {avg_steps:.1f}")
                print(f"  Avg Correct: {avg_correct:.1f}")
                print(f"  Avg Wrong: {avg_wrong:.1f}")
                print(f"  Avg Repeated: {avg_repeated:.2f}")
                print(f"  Q-table Size: {len(self.q_table)} states")
        
        pbar.close()
        
        # Final batch processing
        self._process_batch_updates()
        
        # Clear cache to free memory
        hmm_cache.clear()
        
        return episode_rewards, episode_metrics
    
    def save_model(self, filepath):
        """
        Save Q-table to file.
        
        Args:
            filepath: Path to save file
        """
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(dict(self.q_table), f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load Q-table from file.
        
        Args:
            filepath: Path to load file
        """
        import pickle
        with open(filepath, 'rb') as f:
            q_table_dict = pickle.load(f)
        
        self.q_table = defaultdict(lambda: defaultdict(float))
        for state_key, actions in q_table_dict.items():
            for action, q_value in actions.items():
                self.q_table[state_key][action] = q_value
        
        print(f"Model loaded from {filepath}")


# Alias for compatibility
QLearningAgent = GPUQLearningAgent
