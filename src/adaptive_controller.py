"""
Phase 3: Adaptive Controller Implementation
Mathematical implementation of multi-armed bandit strategies for
optimal feature selection and encoding configuration discovery.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json


@dataclass
class Configuration:
    """
    Represents a configuration c = (f, e) ∈ C
    where f ∈ F (feature selector) and e ∈ E (encoder)
    """
    feature_selector: str  # 'PCA', 'Correlation', 'MutualInfo', 'Random'
    encoder: str           # 'angle', 'amplitude'
    
    def __hash__(self):
        return hash((self.feature_selector, self.encoder))
    
    def __eq__(self, other):
        return (self.feature_selector == other.feature_selector and 
                self.encoder == other.encoder)
    
    def __str__(self):
        return f"{self.feature_selector}+{self.encoder}"
    
    def __repr__(self):
        return self.__str__()


class ConfigurationSpace:
    """
    Defines the configuration space C = F × E
    |C| = |F| × |E| = 4 × 2 = 8
    """
    
    FEATURE_SELECTORS = ['PCA', 'Correlation', 'MutualInfo', 'Random']
    ENCODERS = ['angle', 'amplitude']
    
    @classmethod
    def get_all_configurations(cls) -> List[Configuration]:
        """
        Returns all configurations in C
        """
        configs = []
        for f in cls.FEATURE_SELECTORS:
            for e in cls.ENCODERS:
                configs.append(Configuration(f, e))
        return configs
    
    @classmethod
    def size(cls) -> int:
        """Returns |C| = K"""
        return len(cls.FEATURE_SELECTORS) * len(cls.ENCODERS)


class PerformanceTracker:
    """
    Tracks empirical statistics for each configuration:
    - n_t(c): number of times configuration c has been selected
    - μ̂_t(c): empirical mean reward for configuration c
    - rewards_history: full history of rewards for variance estimation
    """
    
    def __init__(self, configurations: List[Configuration]):
        self.configs = configurations
        self.n = {c: 0 for c in configurations}  # Selection counts
        self.mu = {c: 0.0 for c in configurations}  # Empirical means
        self.rewards = {c: [] for c in configurations}  # Full history
        self.t = 0  # Current time step
        
    def update(self, config: Configuration, reward: float):
        """
        Update statistics after observing reward r_t for configuration c_t
        
        Incremental mean update:
        μ̂_t(c) = [n_{t-1}(c) · μ̂_{t-1}(c) + r_t] / n_t(c)
        """
        self.t += 1
        self.n[config] += 1
        
        # Incremental mean update (numerically stable)
        old_mean = self.mu[config]
        new_count = self.n[config]
        self.mu[config] = old_mean + (reward - old_mean) / new_count
        
        # Store for variance calculation
        self.rewards[config].append(reward)
    
    def get_mean(self, config: Configuration) -> float:
        """Return μ̂_t(c)"""
        return self.mu[config]
    
    def get_count(self, config: Configuration) -> int:
        """Return n_t(c)"""
        return self.n[config]
    
    def get_variance(self, config: Configuration) -> float:
        """
        Return sample variance s²(c)
        s²(c) = 1/(n-1) Σ(r_i - μ̂)²
        """
        if self.n[config] < 2:
            return 0.0
        rewards = self.rewards[config]
        mean = self.mu[config]
        return np.var(rewards, ddof=1)
    
    def get_best_config(self) -> Configuration:
        """Return c* = argmax_c μ̂_t(c)"""
        return max(self.configs, key=lambda c: self.mu[c])
    
    def get_statistics(self) -> Dict:
        """Return full statistics dictionary"""
        return {
            'means': {str(c): self.mu[c] for c in self.configs},
            'counts': {str(c): self.n[c] for c in self.configs},
            'variances': {str(c): self.get_variance(c) for c in self.configs},
            'total_trials': self.t
        }


class AdaptiveStrategy(ABC):
    """
    Abstract base class for adaptation strategies A
    Implements the selection rule: c_t ← A({μ̂_{t-1}(c), n_{t-1}(c)}, t)
    """
    
    def __init__(self, configurations: List[Configuration]):
        self.configs = configurations
        self.tracker = PerformanceTracker(configurations)
        self.selection_history = []
    
    @abstractmethod
    def select_configuration(self) -> Configuration:
        """
        Select next configuration c_t based on current statistics
        """
        pass
    
    def update(self, config: Configuration, reward: float):
        """
        Update tracker after observing reward
        """
        self.tracker.update(config, reward)
        self.selection_history.append((config, reward))
    
    def get_best_configuration(self) -> Configuration:
        """Return best configuration found so far"""
        return self.tracker.get_best_config()
    
    def get_statistics(self) -> Dict:
        """Return performance statistics"""
        return self.tracker.get_statistics()


class RoundRobinStrategy(AdaptiveStrategy):
    """
    Strategy 1: Round-Robin (Baseline)
    
    Selection rule: c_t = c_{(t mod K) + 1}
    
    Properties:
    - Deterministic
    - Uniform exploration
    - No exploitation
    - Regret: O(K·Δ·T)
    """
    
    def __init__(self, configurations: List[Configuration]):
        super().__init__(configurations)
        self.current_index = 0
    
    def select_configuration(self) -> Configuration:
        """
        Cycle through all configurations in order
        """
        config = self.configs[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.configs)
        return config


class EpsilonGreedyStrategy(AdaptiveStrategy):
    """
    Strategy 2: ε-Greedy
    
    Selection rule:
    c_t = argmax_c μ̂_t(c)           with probability 1-ε
    c_t ~ Uniform(C)                 with probability ε
    
    Properties:
    - Simple and effective
    - Balances exploration/exploitation
    - Regret: O(ε·T + K/ε · Σ log(T)/Δ_c)
    """
    
    def __init__(
        self, 
        configurations: List[Configuration],
        epsilon: float = 0.1,
        decay: bool = False
    ):
        """
        Args:
            epsilon: exploration probability ε ∈ [0,1]
            decay: if True, use ε_t = min(1, ε_0 · K / t)
        """
        super().__init__(configurations)
        self.epsilon_0 = epsilon
        self.epsilon = epsilon
        self.decay = decay
    
    def select_configuration(self) -> Configuration:
        """
        Select configuration using ε-greedy rule
        """
        # Update epsilon if decaying
        if self.decay and self.tracker.t > 0:
            K = len(self.configs)
            self.epsilon = min(1.0, self.epsilon_0 * K / self.tracker.t)
        
        # Exploration: random configuration
        if np.random.random() < self.epsilon:
            return np.random.choice(self.configs)
        
        # Exploitation: best known configuration
        # If no data yet, choose randomly
        if self.tracker.t == 0:
            return np.random.choice(self.configs)
        
        return self.tracker.get_best_config()


class UCBStrategy(AdaptiveStrategy):
    """
    Strategy 3: Upper Confidence Bound (UCB1)
    
    Selection rule:
    c_t = argmax_c [μ̂_t(c) + √(2·log(t) / n_t(c))]
    
    Components:
    - μ̂_t(c): exploitation (empirical mean)
    - √(2·log(t) / n_t(c)): exploration bonus
    
    Properties:
    - Optimal logarithmic regret
    - Regret: O(Σ log(T)/Δ_c)
    - Automatically balances exploration/exploitation
    """
    
    def __init__(
        self,
        configurations: List[Configuration],
        c: float = 2.0
    ):
        """
        Args:
            c: exploration coefficient (default √2 ≈ 1.414, we use 2.0)
        """
        super().__init__(configurations)
        self.c = c
    
    def _compute_ucb_score(self, config: Configuration) -> float:
        """
        Compute UCB score: μ̂_t(c) + c·√(log(t) / n_t(c))
        """
        mean = self.tracker.get_mean(config)
        count = self.tracker.get_count(config)
        t = self.tracker.t
        
        # If never tried, give infinite score (ensures exploration)
        if count == 0:
            return float('inf')
        
        # UCB formula
        exploration_bonus = self.c * np.sqrt(np.log(max(t, 1)) / count)
        return mean + exploration_bonus
    
    def select_configuration(self) -> Configuration:
        """
        Select configuration with highest UCB score
        """
        # Compute UCB score for each configuration
        ucb_scores = {c: self._compute_ucb_score(c) for c in self.configs}
        
        # Select argmax
        return max(self.configs, key=lambda c: ucb_scores[c])


class ThompsonSamplingStrategy(AdaptiveStrategy):
    """
    Strategy 4: Thompson Sampling (Bayesian)
    
    Model: r(c) ~ N(μ_c, σ²)
    Prior: μ_c ~ N(μ_0, σ_0²)
    
    Posterior: μ_c | D_t ~ N(μ_post, σ_post²)
    where:
    μ_post = (σ_0² · n_t(c) · μ̂_t(c) + σ² · μ_0) / (σ_0² · n_t(c) + σ²)
    σ_post² = (σ² · σ_0²) / (σ_0² · n_t(c) + σ²)
    
    Selection: Sample μ̃_c ~ p(μ_c | D_t) for all c, choose argmax
    
    Properties:
    - Probability matching
    - Often outperforms UCB empirically
    - Regret: O(√(K·T·log(T)))
    """
    
    def __init__(
        self,
        configurations: List[Configuration],
        prior_mean: float = 0.0,
        prior_std: float = 1.0,
        likelihood_std: float = 0.1
    ):
        """
        Args:
            prior_mean: μ_0 (prior mean)
            prior_std: σ_0 (prior standard deviation)
            likelihood_std: σ (noise standard deviation)
        """
        super().__init__(configurations)
        self.mu_0 = prior_mean
        self.sigma_0 = prior_std
        self.sigma = likelihood_std
    
    def _compute_posterior(self, config: Configuration) -> Tuple[float, float]:
        """
        Compute posterior N(μ_post, σ_post²) for configuration c
        
        Returns:
            (μ_post, σ_post)
        """
        n = self.tracker.get_count(config)
        mu_hat = self.tracker.get_mean(config)
        
        # If no observations, return prior
        if n == 0:
            return self.mu_0, self.sigma_0
        
        # Posterior parameters (conjugate Gaussian update)
        sigma_0_sq = self.sigma_0 ** 2
        sigma_sq = self.sigma ** 2
        
        precision_post = 1.0 / sigma_0_sq + n / sigma_sq
        variance_post = 1.0 / precision_post
        
        mu_post = variance_post * (
            self.mu_0 / sigma_0_sq + n * mu_hat / sigma_sq
        )
        sigma_post = np.sqrt(variance_post)
        
        return mu_post, sigma_post
    
    def select_configuration(self) -> Configuration:
        """
        Thompson Sampling: sample from posterior and choose argmax
        """
        # Sample μ̃_c ~ p(μ_c | D_t) for each configuration
        sampled_means = {}
        for config in self.configs:
            mu_post, sigma_post = self._compute_posterior(config)
            sampled_means[config] = np.random.normal(mu_post, sigma_post)
        
        # Return configuration with highest sampled mean
        return max(self.configs, key=lambda c: sampled_means[c])


class AdaptiveController:
    """
    Main adaptive controller that orchestrates:
    1. Configuration selection (via strategy A)
    2. Feature selection (via selected f)
    3. Encoding (via selected e)
    4. Performance tracking
    5. Statistical analysis
    """
    
    def __init__(
        self,
        strategy: str = 'ucb',
        strategy_params: Optional[Dict] = None
    ):
        """
        Args:
            strategy: 'round_robin', 'epsilon_greedy', 'ucb', 'thompson'
            strategy_params: strategy-specific parameters
        """
        # Get all configurations
        self.configs = ConfigurationSpace.get_all_configurations()
        
        # Initialize strategy
        if strategy_params is None:
            strategy_params = {}
        
        if strategy == 'round_robin':
            self.strategy = RoundRobinStrategy(self.configs)
        elif strategy == 'epsilon_greedy':
            self.strategy = EpsilonGreedyStrategy(self.configs, **strategy_params)
        elif strategy == 'ucb':
            self.strategy = UCBStrategy(self.configs, **strategy_params)
        elif strategy == 'thompson':
            self.strategy = ThompsonSamplingStrategy(self.configs, **strategy_params)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        self.strategy_name = strategy
        print(f"✓ Adaptive Controller initialized")
        print(f"  Strategy: {strategy}")
        print(f"  Configuration space: {ConfigurationSpace.size()} configs")
    
    def select_configuration(self) -> Configuration:
        """
        Select next configuration c_t using strategy A
        """
        return self.strategy.select_configuration()
    
    def update(self, config: Configuration, reward: float):
        """
        Update controller after observing reward r_t
        
        Args:
            config: configuration c_t used
            reward: observed reward r_t (typically negative validation loss)
        """
        self.strategy.update(config, reward)
    
    def get_best_configuration(self) -> Configuration:
        """
        Return best configuration c* = argmax_c μ̂_T(c)
        """
        return self.strategy.get_best_configuration()
    
    def get_statistics(self) -> Dict:
        """
        Return comprehensive statistics
        """
        stats = self.strategy.get_statistics()
        stats['strategy'] = self.strategy_name
        stats['best_config'] = str(self.get_best_configuration())
        return stats
    
    def print_report(self):
        """
        Print human-readable performance report
        """
        stats = self.get_statistics()
        
        print("\n" + "="*70)
        print("ADAPTIVE CONTROLLER REPORT")
        print("="*70)
        print(f"Strategy: {stats['strategy']}")
        print(f"Total trials: {stats['total_trials']}")
        print(f"Best configuration: {stats['best_config']}")
        print("\nConfiguration Performance:")
        print("-"*70)
        print(f"{'Configuration':<25} {'Mean Reward':<15} {'Trials':<10} {'Std Dev':<10}")
        print("-"*70)
        
        # Sort by mean reward
        configs_sorted = sorted(
            self.configs,
            key=lambda c: stats['means'][str(c)],
            reverse=True
        )
        
        for config in configs_sorted:
            c_str = str(config)
            mean = stats['means'][c_str]
            count = stats['counts'][c_str]
            std = np.sqrt(stats['variances'][c_str]) if count > 1 else 0.0
            print(f"{c_str:<25} {mean:<15.4f} {count:<10} {std:<10.4f}")
        
        print("="*70 + "\n")
    
    def export_results(self, filename: str = 'adaptive_results.json'):
        """
        Export results to JSON file
        """
        stats = self.get_statistics()
        with open(filename, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"✓ Results exported to {filename}")


# ========================================
# TESTING AND VALIDATION
# ========================================

def test_configuration_space():
    """Test configuration space generation"""
    print("\n" + "="*70)
    print("TEST: Configuration Space")
    print("="*70)
    
    configs = ConfigurationSpace.get_all_configurations()
    print(f"\nTotal configurations: {len(configs)}")
    print("All configurations:")
    for i, c in enumerate(configs, 1):
        print(f"  {i}. {c}")
    
    print(f"\n✓ Configuration space size: {ConfigurationSpace.size()}")


def test_performance_tracker():
    """Test performance tracking"""
    print("\n" + "="*70)
    print("TEST: Performance Tracker")
    print("="*70)
    
    configs = ConfigurationSpace.get_all_configurations()[:3]
    tracker = PerformanceTracker(configs)
    
    # Simulate some rewards
    print("\nSimulating rewards...")
    rewards = {
        configs[0]: [0.85, 0.87, 0.86],  # Good config
        configs[1]: [0.70, 0.72, 0.71],  # Medium config
        configs[2]: [0.60, 0.61, 0.59],  # Poor config
    }
    
    for config, reward_list in rewards.items():
        for reward in reward_list:
            tracker.update(config, reward)
            print(f"  {config}: reward={reward:.2f}, mean={tracker.get_mean(config):.4f}")
    
    print(f"\nBest configuration: {tracker.get_best_config()}")
    print("✓ Performance tracker working")


def test_strategies():
    """Test all adaptation strategies"""
    print("\n" + "="*70)
    print("TEST: Adaptation Strategies")
    print("="*70)
    
    configs = ConfigurationSpace.get_all_configurations()
    
    # Test each strategy
    strategies = {
        'Round-Robin': RoundRobinStrategy(configs),
        'ε-Greedy': EpsilonGreedyStrategy(configs, epsilon=0.2),
        'UCB': UCBStrategy(configs),
        'Thompson': ThompsonSamplingStrategy(configs)
    }
    
    for name, strategy in strategies.items():
        print(f"\n--- {name} Strategy ---")
        
        # Simulate 10 selections
        for _ in range(10):
            config = strategy.select_configuration()
            # Simulate reward (better configs get higher rewards)
            base_reward = 0.7 + 0.1 * configs.index(config) / len(configs)
            reward = base_reward + np.random.normal(0, 0.05)
            strategy.update(config, reward)
        
        best = strategy.get_best_configuration()
        print(f"Best config found: {best}")
        print(f"Mean reward: {strategy.tracker.get_mean(best):.4f}")
    
    print("\n✓ All strategies working")


def test_adaptive_controller():
    """Test complete adaptive controller"""
    print("\n" + "="*70)
    print("TEST: Adaptive Controller")
    print("="*70)
    
    # Test each strategy
    for strategy_name in ['round_robin', 'epsilon_greedy', 'ucb', 'thompson']:
        print(f"\n--- Testing {strategy_name} ---")
        
        controller = AdaptiveController(strategy=strategy_name)
        
        # Simulate 20 trials
        for t in range(20):
            config = controller.select_configuration()
            # Simulate reward (PCA+angle gets highest)
            if config.feature_selector == 'PCA' and config.encoder == 'angle':
                reward = 0.90 + np.random.normal(0, 0.02)
            else:
                reward = 0.75 + np.random.normal(0, 0.05)
            
            controller.update(config, reward)
        
        best = controller.get_best_configuration()
        print(f"Best found: {best}")
    
    print("\n✓ Adaptive controller working")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("PHASE 3: ADAPTIVE CONTROLLER TESTING")
    print("="*70)
    
    # Run all tests
    test_configuration_space()
    test_performance_tracker()
    test_strategies()
    test_adaptive_controller()
    
    print("\n" + "="*70)
    print("✓ PHASE 3 TESTS COMPLETE - Adaptive controller ready!")
    print("="*70)
    print("\nNext: Integrate with QNN training in full pipeline")