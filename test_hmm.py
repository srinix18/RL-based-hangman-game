"""Quick test of the new ContextualHMM implementation"""
from hmm_model import ContextualHMM

print("Testing Contextual HMM...")
print("=" * 60)

# Initialize and train
hmm = ContextualHMM(corpus_path="corpus.txt", alpha=1.0)
hmm.train(verbose=True)

print("\n" + "=" * 60)
print("Testing predictions...")
print("=" * 60)

# Test prediction
pattern = "_A__E"
guessed = {'A', 'E'}
probs = hmm.predict_letter_probabilities(pattern, guessed)

print(f"\nPattern: {pattern}")
print(f"Guessed: {guessed}")
print(f"Sum of probabilities: {sum(probs):.6f}")
print(f"Entropy: {hmm.entropy(probs):.3f} bits")

# Show top 5 predictions
letters = hmm.letters
top_indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:5]
print(f"\nTop 5 predictions:")
for i in top_indices:
    print(f"  {letters[i]}: {probs[i]:.4f}")

# Test best guess
best = hmm.get_best_guess(pattern, guessed)
print(f"\nBest guess: {best}")

print("\n" + "=" * 60)
print("✓ Test completed successfully!")
print("=" * 60)
