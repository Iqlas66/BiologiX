import numpy as np


class StandardBiologicalLayer:
    """
    Standard Leaky Integrate-and-Fire (LIF) neural layer using
    textbook neuroscience parameters (tau = 20ms time constant).
    """

    def __init__(self, num_neurons, threshold=1.0, time_constant_ms=20.0,refractory_ms = 2.0,a_plus=0.01,a_minus=0.0105,tau_stdp_ms=20.0,w_max=1.5):
        self.num_neurons = num_neurons
        self.threshold = threshold #minimum voltage for activation
        self.leak = 1.0 / time_constant_ms
        self.refractory_ms = refractory_ms

        # Membrane potentials (voltages) for all neurons in the layer
        self.potentials = np.zeros(num_neurons)

        self.a_plus = a_plus
        self.a_minus = a_minus
        self.stdp_decay = np.exp(-1.0/tau_stdp_ms)
        self.w_max = w_max

        self.potentials = np.zeros(num_neurons)
        self.refractory_timer = np.zeros(num_neurons)
        self.trace = np.zeros(num_neurons)
        # Synaptic weight matrix: rows = target neurons, cols = source neurons
        # Positive = Excitatory ("Go"), Negative = Inhibitory ("Stop")
        self.weights = np.zeros((num_neurons, num_neurons))



    def update(self, external_inputs):
        """Simulates 1 millisecond of brain network activity."""

        # 1. Natural Voltage Decay (Leak channels constantly draining charge)
        self.potentials -= self.potentials * self.leak

        # 2. External Sensory Input (Incoming signals from the outside world)
        self.potentials += external_inputs

        self.refractory_timer = np.maximum(self.refractory_timer - 1.0,0.0)

        can_fire = self.refractory_timer <= 0
        spikes = (self.potentials >= self.threshold) & can_fire

        # 4. Synaptic Transmission (Firing neurons pass chemical signals to neighbors)
        if np.any(spikes):
            synaptic_currents = np.dot(self.weights, spikes.astype(float))
            self.potentials += synaptic_currents

        self.stdpr_learn(spikes)


        return spikes
    def stdpr_learn(self, spikes):
        if not np.any(spikes):
            return
        for  i in np.where(spikes)[0]:
            self.trace[i] = 1.0
            self.weights[i,:] += self.a_plus * self.trace * (self.w_max - self.weights[i,:])
            self.weights[:,i] -= self.a_minus *self.trace * self.weights[:,i]
            np.fill_diagonal(self.weights,0.0)


if __name__ == '__main__':
    rng = np.random.default_rng(1)

    layer = StandardBiologicalLayer(num_neurons=3, threshold=1.0, refractory_ms=2.0)


    layer.weights = rng.uniform(0.2, 0.8, size=(3, 3))
    np.fill_diagonal(layer.weights, 0.0)

    print("Initial weights:\n", np.round(layer.weights, 3))


    for t in range(5000):
        stim = np.array([1.5, 0.0, 0.0]) if (t % 50) < 20 else np.zeros(3)
        stim += rng.normal(0, 0.05, size=3)
        layer.update(stim)

    print("\nLearned weights:\n", np.round(layer.weights, 3))