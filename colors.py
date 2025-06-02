import numpy as np
from collections import Counter

# -------------------------------------------------------------
# QuantumColour class (|red>, |green>, |blue>) in a 3-D Hilbert space
# -------------------------------------------------------------
class QuantumColour:
    def __init__(self, amplitudes=None, random_state=None):
        """
        amplitudes : length-3 array-like of complex numbers
                     (order = [red, green, blue])
                     If None, default to |red>
        """
        self.basis = ['red', 'green', 'blue']
        self.n = 3
        self.rng = np.random.default_rng(random_state)

        if amplitudes is None:
            amps = np.array([1, 0, 0], dtype=complex)  # |red>
        else:
            amps = np.asarray(amplitudes, dtype=complex)
            if amps.size != self.n:
                raise ValueError("Need exactly 3 amplitudes (red, green, blue)")
        # normalise
        self.state = amps / np.linalg.norm(amps)

    # ---------- diagnostics ----------
    def amplitudes(self):
        return self.state

    def probs(self):
        """Born-rule probabilities for each colour."""
        return np.abs(self.state) ** 2

    def describe(self, decimals=3):
        lines = []
        for colour, a, p in zip(self.basis, self.state, self.probs()):
            lines.append(
                f"{colour:<5}  amp = {a.real:+.{decimals}f}{a.imag:+.{decimals}f}j   "
                f"P = {p:.{decimals}f}"
            )
        return "\n".join(lines)

    # ---------- quantum ops ----------
    def apply_unitary(self, U):
        """Apply a 3×3 unitary matrix."""
        U = np.asarray(U, dtype=complex)
        if U.shape != (3, 3):
            raise ValueError("U must be 3×3")
        self.state = U @ self.state   # stays normalised

    def set_state(self, amplitudes):
        """Reset the quantum state (auto-normalise)."""
        amps = np.asarray(amplitudes, dtype=complex)
        self.state = amps / np.linalg.norm(amps)

    # ---------- measurement ----------
    def measure(self):
        """
        Collapse to one colour.
        Returns the colour string actually observed.
        """
        idx = self.rng.choice(self.n, p=self.probs())
        outcome = self.basis[idx]
        collapse = np.zeros(self.n, dtype=complex)
        collapse[idx] = 1.0
        self.state = collapse              # post-measurement state
        return outcome

    def sample(self, shots=1000):
        """
        Perform 'shots' independent measurements (without
        permanently collapsing each time).
        Returns Counter with frequencies.
        """
        original_state = self.state.copy()
        counts = Counter()
        for _ in range(shots):
            outcome = self.measure()
            counts[outcome] += 1
            self.state = original_state.copy()  # restore
        return counts


# -------------------------------------------------------------
# Demo
# -------------------------------------------------------------
if __name__ == "__main__":

    # 1. Pure blue ------------------------------------------------
    pure_blue = QuantumColour([0, 0, 1])
    print("== Pure blue state ==")
    print(pure_blue.describe(), end="\n\n")

    # 2. Quantum purple  (|red> + |blue>)/sqrt(2) -----------------
    purple = QuantumColour([1, 0, 1])          # auto-normalised
    print("== Quantum purple  (red + blue) / √2 ==")
    print(purple.describe())
    print("Sampling 20 measurements:", purple.sample(20), end="\n\n")

    # 3. “White”  equal superposition of all three colours -------
    white = QuantumColour([1, 1, 1])
    print("== Quantum white (equal RGB) ==")
    print(white.describe())
    print("Sampling 60 measurements:", white.sample(60), end="\n\n")

    # 4. Apply a custom unitary (here: permutation R→G→B→R) ------
    permute = np.array([[0,1,0],
                        [0,0,1],
                        [1,0,0]], dtype=complex)   # cyclic permutation matrix
    white.apply_unitary(permute)
    print("== After applying permutation unitary to white ==")
    print(white.describe(), end="\n\n")

    # 5. One actual collapse
    colour_observed = white.measure()
    print(f"Measurement collapses to: {colour_observed.upper()}")
    print("Post-measurement state:")
    print(white.describe())