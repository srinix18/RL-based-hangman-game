"""
Main pipeline: Train HMM, Train RL Agent, Evaluate
"""
import numpy as np
import matplotlib.pyplot as plt
from hmm_model import HangmanHMM
from hangman_env import HangmanEnv
from rl_agent import QLearningAgent
from utils import load_words


def train_hmm(corpus_file):
    """Train HMM model on corpus."""
    hmm = HangmanHMM()
    hmm.train(corpus_file)
    return hmm


def train_rl_agent(hmm, train_words, num_episodes=2000, use_gpu=True):
    """Train RL agent on Hangman environment (reduced episodes for speed)."""
    print(f"\nTraining RL agent for {num_episodes} episodes (optimized for hackathon speed)...")
    env = HangmanEnv(train_words)
    agent = QLearningAgent(hmm, use_gpu=use_gpu)
    episode_rewards, episode_metrics = agent.train(env, num_episodes=num_episodes, verbose=True, log_interval=200)
    return agent, episode_rewards, episode_metrics


def evaluate_agent(agent, test_words):
    """
    Evaluate agent on test set.
    
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\nEvaluating agent on {len(test_words)} test words...")
    
    env = HangmanEnv(test_words)
    wins = 0
    total_wrong_guesses = 0
    total_repeated_guesses = 0
    
    for i, word in enumerate(test_words):
        state = env.reset(word=word)
        game_wrong = 0
        game_repeated = 0
        
        while not state['done']:
            available_actions = env.get_available_actions()
            action = agent.choose_action(state, available_actions, training=False)
            
            next_state, reward, done, info = env.step(action)
            
            if info['repeated']:
                game_repeated += 1
            elif not info['correct']:
                game_wrong += 1
            
            state = next_state
        
        # Check if won
        if state['lives'] > 0:
            wins += 1
        
        total_wrong_guesses += game_wrong
        total_repeated_guesses += game_repeated
        
        if (i + 1) % 100 == 0:
            print(f"Evaluated {i + 1}/{len(test_words)} words...")
    
    success_rate = wins / len(test_words)
    avg_wrong = total_wrong_guesses / len(test_words)
    avg_repeated = total_repeated_guesses / len(test_words)
    
    # Calculate final score
    final_score = (success_rate * 2000) - (total_wrong_guesses * 5) - (total_repeated_guesses * 2)
    
    metrics = {
        'success_rate': success_rate,
        'wins': wins,
        'total_games': len(test_words),
        'total_wrong_guesses': total_wrong_guesses,
        'total_repeated_guesses': total_repeated_guesses,
        'avg_wrong_guesses': avg_wrong,
        'avg_repeated_guesses': avg_repeated,
        'final_score': final_score
    }
    
    return metrics


def plot_learning_curve(episode_rewards, window=100):
    """Plot and save learning curve."""
    # Calculate moving average
    moving_avg = []
    for i in range(len(episode_rewards)):
        start = max(0, i - window + 1)
        moving_avg.append(np.mean(episode_rewards[start:i+1]))
    
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards, alpha=0.3, label='Episode Reward')
    plt.plot(moving_avg, label=f'Moving Average ({window} episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('RL Agent Learning Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('learning_curve.png', dpi=300, bbox_inches='tight')
    print("\nLearning curve saved as 'learning_curve.png'")
    plt.close()


def save_report(metrics, filename='analysis_report.txt'):
    """Save evaluation report to file."""
    with open(filename, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("HANGMAN AI - EVALUATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Total Games: {metrics['total_games']}\n")
        f.write(f"Games Won: {metrics['wins']}\n")
        f.write(f"Success Rate: {metrics['success_rate']:.2%}\n\n")
        
        f.write(f"Total Wrong Guesses: {metrics['total_wrong_guesses']}\n")
        f.write(f"Average Wrong Guesses per Game: {metrics['avg_wrong_guesses']:.2f}\n\n")
        
        f.write(f"Total Repeated Guesses: {metrics['total_repeated_guesses']}\n")
        f.write(f"Average Repeated Guesses per Game: {metrics['avg_repeated_guesses']:.2f}\n\n")
        
        f.write("=" * 60 + "\n")
        f.write(f"FINAL SCORE: {metrics['final_score']:.2f}\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("Score Formula:\n")
        f.write("  (Success Rate * 2000) - (Total Wrong * 5) - (Total Repeated * 2)\n\n")
        
        f.write("Key Observations:\n")
        f.write(f"- Win rate of {metrics['success_rate']:.1%} demonstrates effective learning\n")
        f.write(f"- Average {metrics['avg_wrong_guesses']:.2f} wrong guesses per game\n")
        f.write(f"- Minimal repeated guesses ({metrics['avg_repeated_guesses']:.2f} per game)\n")
    
    print(f"\nAnalysis report saved as '{filename}'")


def main():
    """Main execution pipeline."""
    print("=" * 60)
    print("HANGMAN AI - END-TO-END INTELLIGENT AGENT")
    print("=" * 60)
    
    # File paths
    corpus_file = 'corpus.txt'
    test_file = 'test.txt'
    
    # Step 1: Train HMM
    print("\n[STEP 1] Training Hidden Markov Model...")
    hmm = train_hmm(corpus_file)
    
    # Step 2: Load training words for RL
    train_words = load_words(corpus_file)
    
    # Step 3: Train RL Agent (reduced to 2000 episodes for speed)
    print("\n[STEP 2] Training Reinforcement Learning Agent...")
    agent, episode_rewards, episode_metrics = train_rl_agent(hmm, train_words, num_episodes=2000)
    
    # Save detailed metrics
    print("\n[STEP 3] Saving training metrics...")
    import json
    with open('training_metrics.json', 'w') as f:
        json.dump(episode_metrics, f, indent=2)
    print("Training metrics saved to training_metrics.json")
    
    # Step 4: Plot learning curve
    print("\n[STEP 4] Generating learning curve...")
    plot_learning_curve(episode_rewards)
    
    # Step 5: Evaluate on test set
    print("\n[STEP 5] Evaluating on test set...")
    test_words = load_words(test_file)
    metrics = evaluate_agent(agent, test_words)
    
    # Step 6: Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total Games: {metrics['total_games']}")
    print(f"Games Won: {metrics['wins']}")
    print(f"Success Rate: {metrics['success_rate']:.2%}")
    print(f"\nTotal Wrong Guesses: {metrics['total_wrong_guesses']}")
    print(f"Average Wrong Guesses per Game: {metrics['avg_wrong_guesses']:.2f}")
    print(f"\nTotal Repeated Guesses: {metrics['total_repeated_guesses']}")
    print(f"Average Repeated Guesses per Game: {metrics['avg_repeated_guesses']:.2f}")
    print(f"\n{'=' * 60}")
    print(f"FINAL SCORE: {metrics['final_score']:.2f}")
    print("=" * 60)
    
    # Step 7: Save report
    print("\n[STEP 6] Saving analysis report...")
    save_report(metrics)
    
    print("\n✓ All tasks completed successfully!")


if __name__ == "__main__":
    main()
