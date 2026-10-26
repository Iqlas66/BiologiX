import numpy as np


class StandardBiologicalLayer:
    """
    Standard Leaky Integrate-and-Fire (LIF) neural layer using
    textbook neuroscience parameters (tau = 20ms time constant).
    """

    def __init__(self, num_neurons, threshold=1.0, time_constant_ms=20.0):
        self.num_neurons = num_neurons
        self.threshold = threshold #minimum voltage for activation
        self.leak = 1.0 / time_constant_ms

        # Membrane potentials (voltages) for all neurons in the layer
        self.potentials = np.zeros(num_neurons)

        # Synaptic weight matrix: rows = target neurons, cols = source neurons
        # Positive = Excitatory ("Go"), Negative = Inhibitory ("Stop")
        self.weights = np.zeros((num_neurons, num_neurons))

    def update(self, external_inputs):
        """Simulates 1 millisecond of brain network activity."""

        # 1. Natural Voltage Decay (Leak channels constantly draining charge)
        self.potentials -= self.potentials * self.leak

        # 2. External Sensory Input (Incoming signals from the outside world)
        self.potentials += external_inputs

        # 3. Threshold Check (Did voltage cross the firing line?)
        spikes = self.potentials >= self.threshold

        # 4. Synaptic Transmission (Firing neurons pass chemical signals to neighbors)
        if np.any(spikes):
            synaptic_currents = np.dot(self.weights, spikes.astype(float))
            self.potentials += synaptic_currents

        # 5. Refractory Reset (Voltage drops back to 0 after firing an action potential)
        self.potentials[spikes] = 0.0

        return spikes


if __name__ == '__main__':
    # Initialize network with standard parameters
    layer = StandardBiologicalLayer(num_neurons=3, threshold=1.0, time_constant_ms=20.0)

    # Standard wiring setup:
    layer.weights[1, 0] = 0.8  # Excitatory connection (Neuron 0 -> Neuron 1)
    layer.weights[2, 0] = 1.0
    layer.weights[0 , 2] = 1.0
    layer.weights[2, 1] = -0.7  # Inhibitory connection (Neuron 1 -> Neuron 2)

    # Standard 5-millisecond stimulus test sequence
    stimulus_timeline = [
        np.array([1.2, 0.0, 0.0]),  # ms 1: Strong stimulus to Neuron 0 (fires immediately)
        np.array([1.2, 0.0, 0.0]),  # ms 2: Let the network cascade
        np.array([0.0, 0.0, 0.0]),  # ms 3: Let it decay
        np.array([0.0, 0.0, 0.0]),  # ms 4: Rest
        np.array([0.0, 0.0, 0.0]),  # ms 5: Rest
    ]

    print("Time (ms) | Potentials [n0, n1, n2]     | Spikes [n0, n1, n2]")
    print("-" * 60)
    for t, inputs in enumerate(stimulus_timeline, 1):
        spikes = layer.update(inputs)
        print(f"   {t:2d}     | {np.round(layer.potentials, 2)}          | {spikes}")