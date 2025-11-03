# RL-based Hangman Game

An end-to-end intelligent Hangman agent using **Machine Learning** that combines Hidden Markov Models (HMM) and Q-Learning (Reinforcement Learning) for optimal letter prediction.

## 🎯 Overview

This project implements a sophisticated Hangman AI that learns to play the game optimally by:

- Using **Hidden Markov Models** for probabilistic letter prediction based on word patterns
- Employing **Q-Learning** (Reinforcement Learning) for decision-making and strategy optimization
- Training on 50,000 words and evaluating on 2,000 test words
- Real-time progress tracking with win rates during training

## 🧩 System Architecture

### 1. Hidden Markov Model (HMM)

- Models letter probability distributions based on word patterns
- Uses frequency-based approach with position-aware statistics
- Filters candidate words matching current game state

### 2. Hangman Environment

- OpenAI Gym-style game environment
- Supports `reset()`, `step()`, and `render()` methods
- Configurable wrong guess limits (default: 6)

### 3. Q-Learning Agent

- Tabular Q-learning with ε-greedy exploration
- Dictionary-based Q-table for state-action values
- Integrates HMM probabilities for smarter action selection
- Real-time progress bar with win rate tracking

**Reward Structure:**

- +10 for correct guess
- -5 for wrong guess
- -2 for repeated guess
- +100 for winning
- -50 for losing

## 📦 Project Structure

```
ml hackathon/
├── corpus.txt          # Training data (50,000 words)
├── test.txt            # Test data (2,000 words)
├── utils.py            # Helper functions
├── hmm_model.py        # HMM implementation
├── hangman_env.py      # Game environment
├── rl_agent.py         # Q-Learning agent
└── main.py             # Main pipeline
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install numpy matplotlib tqdm
```

Or install all dependencies:

```bash
pip install -r requirements.txt
```

### Run the Complete Pipeline

```bash
python main.py
```

This will:

1. Train HMM on `corpus.txt`
2. Train Q-Learning agent for 5,000 episodes with progress bar
3. Evaluate on `test.txt`
4. Generate `learning_curve.png` and `analysis_report.txt`

## 📊 Evaluation Metrics

The system computes:

- **Success Rate**: Percentage of games won
- **Average Wrong Guesses**: Per game
- **Average Repeated Guesses**: Per game
- **Final Score**: `(Success Rate × 2000) - (Total Wrong × 5) - (Total Repeated × 2)`

## 🧠 Key Features

✅ HMM-based letter probability prediction  
✅ Q-Learning with epsilon-greedy exploration  
✅ Smart integration of HMM probabilities into RL decisions  
✅ Real-time progress bar with win rate tracking  
✅ Comprehensive evaluation and visualization  
✅ Modular, extensible architecture

## 📈 Output Files

- `learning_curve.png` - Training progress visualization
- `analysis_report.txt` - Detailed performance metrics

## 🛠️ Technical Details

- **Language**: Python 3.12+
- **ML Techniques**: Hidden Markov Models, Q-Learning
- **Libraries**: NumPy, Matplotlib, Collections

## 📝 Algorithm Flow

1. **HMM Training**: Learn letter frequencies and patterns from corpus
2. **RL Training**: Agent plays thousands of games, learning optimal strategies
3. **Evaluation**: Test on unseen words and compute performance metrics
4. **Visualization**: Generate learning curves and analysis reports

## 🎓 Future Enhancements

- Implement Deep Q-Network (DQN) for neural network-based learning
- Add support for variable difficulty levels
- Implement more sophisticated state representations
- Explore other RL algorithms (A3C, PPO)

## 📄 License

This project is open source and available for educational purposes.

---

**Built with ❤️ using Machine Learning and Reinforcement Learning**
