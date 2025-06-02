import numpy as np
from scipy.stats import unitary_group

# ----- Define the QuantumHuman class -----
class QuantumHuman:
    def __init__(self, amplitudes=None, random_state=None):
        self.emotions = ['joy', 'sadness', 'anger', 'fear', 'disgust',
                         'surprise', 'trust', 'anticipation', 'acceptance', 'boredom']
        self.n = len(self.emotions)
        self.rng = np.random.default_rng(random_state)
        # Start: equal superposition unless specified
        if amplitudes is not None:
            arr = np.array(amplitudes, dtype=complex)
            arr = arr / np.linalg.norm(arr)
            self.state = arr
        else:
            arr = np.ones(self.n, dtype=complex) / np.sqrt(self.n)
            self.state = arr

    def get_probabilities(self):
        """Probability for each emotion upon measurement."""
        return np.abs(self.state) ** 2

    def measure(self):
        """Collapse to one emotion, return emotion name and final probabilities."""
        probs = self.get_probabilities()
        idx = self.rng.choice(self.n, p=probs)
        collapse = np.zeros(self.n, dtype=complex)
        collapse[idx] = 1.0
        self.state = collapse
        return self.emotions[idx], probs
    
    def apply_unitary(self, U):
        """Apply a 10x10 unitary to the state vector"""
        self.state = U @ self.state
        # Should remain normalized if U is a true unitary

    def set_state(self, amplitudes):
        arr = np.array(amplitudes, dtype=complex)
        self.state = arr / np.linalg.norm(arr)

    def describe_state(self, decimals=3):
        amp = self.state
        probs = self.get_probabilities()
        out = []
        for em, a, p in zip(self.emotions, amp, probs):
            out.append(f"{em.title():<12}: amplitude = {a.real:+.{decimals}f}{a.imag:+.{decimals}f}j, pr = {p:.3f}")
        return '\n'.join(out)

# ----- Plotlines dictionary -----
plotlines = {
    'joy': 'Plot 1: A joyous reunion changes everything.',
    'sadness': 'Plot 2: A loss forms the crux of the story.',
    'anger': 'Plot 3: Conflict escalates between rivals.',
    'fear': 'Plot 4: The town faces an unknown threat.',
    'disgust': 'Plot 5: Betrayal shakes the community.',
    'surprise': 'Plot 6: An unexpected visitor arrives.',
    'trust': 'Plot 7: Allies form and secrets are shared.',
    'anticipation': 'Plot 8: The finale is teased!',
    'acceptance': 'Plot 9: Characters learn to forgive.',
    'boredom': 'Plot 10: Nothing much happens... yet.'
}
emotions = list(plotlines.keys())

# ----- 1. Prepare your QuantumHuman in emotional superposition -----
qh = QuantumHuman()
print("Initial quantum superposition of viewer's emotions:")
print(qh.describe_state())

# (Optional) Apply a random emotional 'disturbance'
U = unitary_group.rvs(10)
qh.apply_unitary(U)  # You can comment this out to leave the state as equal superposition
print("\nAfter random emotional quantum operation:")
print(qh.describe_state())

# ----- 2. Measure emotion -----
print("\nMeasuring emotional state...")
actual_emotion, previous_probs = qh.measure()
print(f"Your emotion upon observing 'Life' is: **{actual_emotion.upper()}**")
print("Probabilities that led to this outcome:")
for em, p in zip(qh.emotions, previous_probs):
    print(f"  {em}: {p:.3f}")

# ----- 3. Use these probabilities to determine plot outcome -----
def measure_show_weighted(prev_probs, plotlines, emotions):
    """Choose plotline using the measured emotional state's probabilistic weights."""
    idx = np.random.choice(len(emotions), p=prev_probs)
    chosen_emotion = emotions[idx]
    plot = plotlines[chosen_emotion]
    return chosen_emotion, plot

final_plot_emotion, plot = measure_show_weighted(previous_probs, plotlines, emotions)
print("\n------- TV Show 'Life' (Quantum Collapsed) -------")
print(f"The show actualizes as: {final_plot_emotion.upper()}")
print(f"You experience: {plot}")