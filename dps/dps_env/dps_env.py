from typing import Dict, List

import gymnasium as gym
from gymnasium import spaces

from dps.utils.precision import PENALTIES, LinkType, Precision


class DPSEnvironment(gym.Env):
    """
    Gym-like environment for training dynamic precision selection policies.
    """

    def __init__(
        self,
        network_simulator,
        model_simulator,
        reward_config,
        precisions: List[str] = list(Precision),
    ):
        super().__init__()

        # Simulators
        self.network_simulator = network_simulator  # TODO: NS-3 based simulator?
        self.model_simulator = (
            model_simulator  # TODO: maybe not needed? Simulates model training
        )

        # Define action space (precision options)
        self.precisions = precisions
        self.action_space = spaces.Discrete(
            len(precisions)
        )  ## TODO: or maybe accept th
        self.precision_map = {i: p for i, p in precisions}

        # Define observation space
        # Includes network stats, tensor properties, training progress
        self.observation_space = spaces.Dict(
            {
                "network": spaces.Dict(
                    {
                        "congestion": spaces.Box(low=0, high=1, shape=(1,)),
                        "bandwidth": spaces.Box(low=0, high=float("inf"), shape=(1,)),
                        "utilization": spaces.Box(low=0, high=1, shape=(1,)),
                    }
                ),
                "tensor": spaces.Dict(
                    {
                        "size": spaces.Box(low=0, high=float("inf"), shape=(1,)),
                        "layer_idx": spaces.Discrete(100),  # Assuming max 100 layers
                        "is_backward": spaces.Discrete(2),  # 0: forward, 1: backward
                    }
                ),
                "training": spaces.Dict(
                    {
                        "progress": spaces.Box(low=0, high=1, shape=(1,)),
                        "accuracy": spaces.Box(low=0, high=1, shape=(1,)),
                        # TODO:  maybe loss?
                    }
                ),
                "topology": spaces.Dict(
                    {
                        "distance": spaces.Box(low=0, high=10, dtype=int),
                        "link_type": spaces.Discrete(
                            len(LinkType)
                        ),  # Type of link (intra-node, inter-node, etc.)
                    }
                ),
            }
        )

        self.reward_calculator = RewardCalculator(reward_config)

        # State tracking
        self.current_step = 0
        self.total_steps = model_simulator.total_steps
        self.state = None
        self.last_action = None
        self.communication_history = []

    def reset(self):
        """Reset environment to beginning of training."""
        self.current_step = 0
        self.model_simulator.reset()
        self.network_simulator.reset()
        self.communication_history = []

        # Get initial state
        self.state = self._get_observation()
        return self.state

    def step(self, action):
        """
        Take a step in the environment by selecting a precision for the current communication.

        Args:
            action: Precision to use (index into precision options)

        Returns:
            observation: Next state
            reward: Reward for the action
            done: Whether episode is complete
            info: Additional information
        """
        # Convert action index to precision
        precision = self._action_to_precision(action)

        # Apply selected precision for current communication
        latency, accuracy_impact = self._apply_precision(precision)

        # Update history
        self.last_action = action
        self.communication_history.append(
            {
                "step": self.current_step,
                "state": self.state,
                "action": action,
                "precision": precision,
                "latency": latency,
                "accuracy_impact": accuracy_impact,
            }
        )

        # Calculate reward
        reward = self.reward_calculator.calculate_reward(
            latency=latency,
            accuracy_impact=accuracy_impact,
            precision=precision,
            state=self.state,
        )

        # Advance simulation
        self.current_step += 1
        done = self.current_step >= self.total_steps

        # Get new state
        self.state = self._get_observation()

        # Additional info
        info = {
            "latency": latency,
            "accuracy_impact": accuracy_impact,
            "communication_count": self.current_step,
        }

        return self.state, reward, done, info

    def _get_observation(self):
        """Get current observation/state from simulators."""
        network_stats = self.network_simulator.get_current_stats()
        model_stats = self.model_simulator.get_current_stats()
        tensor_info = self.model_simulator.get_current_tensor_info()
        topology_info = self.network_simulator.get_topology_info_for_current_comm()

        return {
            "network": network_stats,
            "tensor": tensor_info,
            "training": {
                "progress": self.current_step / self.total_steps,
                "accuracy": model_stats["current_accuracy"],
            },
            "topology": topology_info,
        }

    def _action_to_precision(self, action):
        """Convert action index to precision type."""
        return self.precision_map[action]

    def _apply_precision(self, precision):
        """Apply selected precision and get resulting latency and accuracy impact."""
        # Simulate communication with selected precision
        latency = self.network_simulator.simulate_communication(
            tensor_info=self.state["tensor"], precision=precision
        )

        # Simulate effect on model
        accuracy_impact = self.model_simulator.simulate_precision_impact(
            tensor_info=self.state["tensor"], precision=precision
        )

        return latency, accuracy_impact


class RewardCalculator:
    """Calculate rewards based on latency and accuracy impact."""

    def __init__(
        self,
        latency_weight: float = 0.7,
        accuracy_weight: float = 0.3,
        precision_penalty: Dict[Precision, float] = PENALTIES,
    ):
        self.latency_weight = latency_weight
        self.accuracy_weight = accuracy_weight
        self.precision_penalty = precision_penalty

    def calculate_reward(self, latency, accuracy_impact, precision, state):
        """
        Calculate reward for a precision selection.

        Higher reward for:
        - Lower latency
        - Higher accuracy
        - Using higher precision during backward pass or early in training
        """
        # Normalize latency (lower is better)
        latency_component = -latency * self.latency_weight

        # Accuracy impact (higher is better)
        accuracy_component = accuracy_impact * self.accuracy_weight

        # Additional penalty for low precision in backward pass
        is_backward = state["tensor"]["is_backward"]
        precision_penalty = self.precision_penalty[precision]
        if is_backward:
            precision_penalty *= 2  # Double penalty in backward pass

        # Additional factors based on training progress
        training_progress = state["training"]["progress"]
        if training_progress < 0.2:  # Early in training
            precision_penalty *= (
                1.5  # Higher penalty for low precision early in training
            )

        return latency_component + accuracy_component + precision_penalty
